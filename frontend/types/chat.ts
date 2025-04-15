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