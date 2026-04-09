import argparse
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from dataset import (
    BertQADataset,
    GloveQADataset,
    Vocab,
    bert_collate_fn,
    glove_collate_fn,
    load_glove_embeddings,
    load_squad,
)
from models import BertBiDAF, GloveBiDAF
from utils import (
    AverageMeter,
    best_span_from_logits,
    compute_em_f1,
    ensure_dir,
    print_device_status,
    save_json,
    save_metrics_txt,
    set_seed,
    str2bool,
)

def safe_masked_fill(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0
    fill_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask, fill_value)

def parse_args():
    parser = argparse.ArgumentParser(description="Train BiDAF reading comprehension models")
    parser.add_argument("--embedding_type", type=str, default="bert", choices=["bert", "glove"])
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_bert", type=str2bool, default=True)
    parser.add_argument("--glove_path", type=str, default="./data/glove/glove.6B.100d.txt")
    parser.add_argument("--train_file", type=str, default="./data/squad/train-v1.1.json")
    parser.add_argument("--test_file", type=str, default="./data/squad/dev-v1.1.json")
    parser.add_argument("--output_dir", type=str, default="./code/task_2/runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--max_question_len", type=int, default=32)
    parser.add_argument("--max_answer_len", type=int, default=30)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_sample_limit", type=int, default=0)
    parser.add_argument("--eval_sample_limit", type=int, default=0)
    parser.add_argument("--train_embeddings", type=str2bool, default=True)
    parser.add_argument("--report_test_each_epoch", type=str2bool, default=True)
    return parser.parse_args()


def decode_predictions_glove(batch, start_idx, end_idx):
    predictions = []
    for tokens, s, e in zip(batch["context_tokens"], start_idx.tolist(), end_idx.tolist()):
        s = max(0, min(s, len(tokens) - 1))
        e = max(s, min(e, len(tokens) - 1))
        predictions.append(" ".join(tokens[s : e + 1]))
    return predictions


def decode_predictions_bert(batch, start_idx, end_idx):
    predictions = []
    for text, offsets, s, e in zip(batch["context_text"], batch["context_offsets"], start_idx.tolist(), end_idx.tolist()):
        if s >= len(offsets):
            predictions.append("")
            continue
        if e >= len(offsets):
            e = len(offsets) - 1
        if offsets[s][0] == offsets[s][1]:
            predictions.append("")
            continue
        start_char = offsets[s][0]
        end_char = offsets[e][1]
        if end_char <= start_char:
            predictions.append("")
        else:
            predictions.append(text[start_char:end_char])
    return predictions


def get_predictions_text(batch, start_idx, end_idx, embedding_type):
    if embedding_type == "glove":
        return decode_predictions_glove(batch, start_idx, end_idx)
    return decode_predictions_bert(batch, start_idx, end_idx)


@torch.no_grad()
def evaluate(model, data_loader, device, embedding_type, max_answer_len=30):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    total_em = 0.0
    total_f1 = 0.0
    total_examples = 0

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        question_ids = batch["question_ids"].to(device)
        question_mask = batch["question_mask"].to(device)
        context_ids = batch["context_ids"].to(device)
        context_mask = batch["context_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        start_logits, end_logits = model(
            context_ids=context_ids,
            question_ids=question_ids,
            context_mask=context_mask,
            question_mask=question_mask,
        )

        if embedding_type == "bert" and "context_span_mask" in batch:
            span_mask = batch["context_span_mask"].to(device)
            start_logits = safe_masked_fill(start_logits, span_mask)
            end_logits = safe_masked_fill(end_logits, span_mask)

        loss = ce_loss(start_logits, start_positions) + ce_loss(end_logits, end_positions)
        loss_meter.update(loss.item(), question_ids.size(0))

        pred_start, pred_end = best_span_from_logits(start_logits, end_logits, max_answer_len=max_answer_len)
        pred_texts = get_predictions_text(batch, pred_start.cpu(), pred_end.cpu(), embedding_type)

        for pred_text, gold_answers in zip(pred_texts, batch["gold_answers"]):
            em, f1 = compute_em_f1(pred_text, gold_answers)
            total_em += em
            total_f1 += f1
            total_examples += 1

    return {
        "loss": loss_meter.avg,
        "em": 100.0 * total_em / max(1, total_examples),
        "f1": 100.0 * total_f1 / max(1, total_examples),
    }


def build_model_and_loaders(args):
    train_examples = load_squad(args.train_file)
    test_examples = load_squad(args.test_file)

    if args.train_sample_limit > 0:
        train_examples = train_examples[: args.train_sample_limit]
    if args.eval_sample_limit > 0:
        test_examples = test_examples[: args.eval_sample_limit]

    print(f"[INFO] Loaded {len(train_examples)} raw training examples.")
    print(f"[INFO] Loaded {len(test_examples)} raw test examples.")

    if args.embedding_type == "glove":
        if not os.path.exists(args.glove_path):
            raise FileNotFoundError(
                f"GloVe file not found at {args.glove_path}. Download glove.6B.100d.txt or run with --embedding_type bert"
            )
        vocab = Vocab.build(train_examples, min_freq=2)
        embedding_matrix = load_glove_embeddings(args.glove_path, vocab, embedding_dim=100)

        full_train_dataset = GloveQADataset(
            train_examples,
            vocab=vocab,
            max_context_len=args.max_context_len,
            max_question_len=args.max_question_len,
        )
        test_dataset = GloveQADataset(
            test_examples,
            vocab=vocab,
            max_context_len=args.max_context_len,
            max_question_len=args.max_question_len,
        )

        model = GloveBiDAF(
            embedding_matrix=embedding_matrix,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            train_embeddings=args.train_embeddings,
        )
        collate_fn = glove_collate_fn

    else:
        full_train_dataset = BertQADataset(
            train_examples,
            model_name=args.bert_model_name,
            max_context_len=args.max_context_len,
            max_question_len=args.max_question_len,
        )
        test_dataset = BertQADataset(
            test_examples,
            model_name=args.bert_model_name,
            max_context_len=args.max_context_len,
            max_question_len=args.max_question_len,
        )

        model = BertBiDAF(
            bert_model_name=args.bert_model_name,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            freeze_bert=args.freeze_bert,
        )
        collate_fn = bert_collate_fn

    val_size = int(len(full_train_dataset) * args.val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"[INFO] Final train size: {len(train_dataset)}")
    print(f"[INFO] Final validation size: {len(val_dataset)}")
    print(f"[INFO] Final test size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return model, train_loader, val_loader, test_loader


def build_optimizer(model, args):
    if args.embedding_type == "bert" and not args.freeze_bert:
        bert_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert"):
                bert_params.append(param)
            else:
                head_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": args.bert_lr},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return optimizer


def main():
    args = parse_args()
    set_seed(args.seed)
    device = print_device_status()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.embedding_type}_bidaf_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(run_dir)

    print(f"[INFO] Run directory: {run_dir}")
    save_json(os.path.join(run_dir, "config.json"), vars(args))

    model, train_loader, val_loader, test_loader = build_model_and_loaders(args)
    model.to(device)

    optimizer = build_optimizer(model, args)
    ce_loss = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    best_val_f1 = -1.0
    best_epoch = -1
    history = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            question_ids = batch["question_ids"].to(device, non_blocking=True)
            question_mask = batch["question_mask"].to(device, non_blocking=True)
            context_ids = batch["context_ids"].to(device, non_blocking=True)
            context_mask = batch["context_mask"].to(device, non_blocking=True)
            start_positions = batch["start_positions"].to(device, non_blocking=True)
            end_positions = batch["end_positions"].to(device, non_blocking=True)

            with amp.autocast(device_type="cuda", enabled=use_amp):
                start_logits, end_logits = model(
                    context_ids=context_ids,
                    question_ids=question_ids,
                    context_mask=context_mask,
                    question_mask=question_mask,
                )
                if args.embedding_type == "bert" and "context_span_mask" in batch:
                    span_mask = batch["context_span_mask"].to(device, non_blocking=True)
                    start_logits = safe_masked_fill(start_logits, span_mask)
                    end_logits = safe_masked_fill(end_logits, span_mask)
                loss = ce_loss(start_logits, start_positions) + ce_loss(end_logits, end_positions)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * args.grad_accum_steps
            train_loss_meter.update(actual_loss, question_ids.size(0))
            progress.set_postfix(train_loss=f"{train_loss_meter.avg:.4f}")

        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            embedding_type=args.embedding_type,
            max_answer_len=args.max_answer_len,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss_meter.avg,
            "val_loss": val_metrics["loss"],
            "val_em": val_metrics["em"],
            "val_f1": val_metrics["f1"],
        }

        if args.report_test_each_epoch:
            epoch_test_metrics = evaluate(
                model,
                test_loader,
                device=device,
                embedding_type=args.embedding_type,
                max_answer_len=args.max_answer_len,
            )
            history_row["test_loss"] = epoch_test_metrics["loss"]
            history_row["test_em"] = epoch_test_metrics["em"]
            history_row["test_f1"] = epoch_test_metrics["f1"]
        history.append(history_row)

        msg = (
            f"[EPOCH {epoch}] train_loss={train_loss_meter.avg:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_em={val_metrics['em']:.2f} | val_f1={val_metrics['f1']:.2f}"
        )
        if args.report_test_each_epoch:
            msg += (
                f" | test_loss={history_row['test_loss']:.4f} | "
                f"test_em={history_row['test_em']:.2f} | test_f1={history_row['test_f1']:.2f}"
            )
        print(msg)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"[INFO] Saved new best model at epoch {epoch}.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    best_model_path = os.path.join(run_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        embedding_type=args.embedding_type,
        max_answer_len=args.max_answer_len,
    )

    summary = {
        "embedding_type": args.embedding_type,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 4),
        "test_loss": round(test_metrics["loss"], 6),
        "test_em": round(test_metrics["em"], 4),
        "test_f1": round(test_metrics["f1"], 4),
    }

    save_json(os.path.join(run_dir, "summary.json"), summary)
    save_metrics_txt(os.path.join(run_dir, "metrics_log.txt"), history, summary)

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best validation F1: {best_val_f1:.2f}")
    print(
        f"[INFO] Test metrics -> loss: {test_metrics['loss']:.4f}, "
        f"EM: {test_metrics['em']:.2f}, F1: {test_metrics['f1']:.2f}"
    )
    print(f"[INFO] Metrics log saved to: {os.path.join(run_dir, 'metrics_log.txt')}")


if __name__ == "__main__":
    main()
