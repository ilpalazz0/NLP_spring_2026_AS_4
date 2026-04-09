const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // ignore JSON parse failure
    }
    throw new Error(detail);
  }

  return response.json();
}

export { API_BASE_URL };

export function fetchHealth() {
  return request('/api/health');
}

export function fetchModels(refresh = false) {
  return request(`/api/models${refresh ? '?refresh=true' : ''}`);
}

export function fetchModelDetail(runId) {
  return request(`/api/models/${runId}`);
}

export function predictAnswer(payload) {
  return request('/api/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function predictSentiment(payload) {
  return request('/api/sentiment/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
