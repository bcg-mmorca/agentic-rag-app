'use server';

import { AssistantResponse } from "@/types/chat";

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function processFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BACKEND_URL}/process`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('File upload failed');
  }

  return response.json();
}

export async function askAssistant(query: string) {
  const response = await fetch(`${BACKEND_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const res = await response.text();
    throw new Error(`Failed to get assistant response: ${res}`);
  }

  const reply = await response.json();
  
  return reply as AssistantResponse;
}
