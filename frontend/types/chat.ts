export interface Message {
  role: 'user' | 'assistant';
  content: string;
  id?: string;
  timestamp?: Date;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface AssistantResponse {
  question: string;
  answer: string;
  source_documents: string[];
}
