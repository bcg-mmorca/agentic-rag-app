'use client';

import { askAssistant, processFile } from '@/actions/api';
import { Message } from '@/types/chat';
import { useState } from 'react';
import ChatMessages from '@/components/ChatMessages';
import ChatInputForm from '@/components/ChatInputForm';
import FileUpload from '@/components/FileUpload';

export default function Home() {
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false); // New state for typing indicator

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      await processFile(file);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `File "${file.name}" has been processed successfully.`
      }]);
    } catch (error) {
      console.error('Upload failed:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Failed to upload file. Please try again.'
      }]);
    } finally {
      setIsUploading(false);
      e.target.value = ''; // Reset file input
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages([...messages, { role: 'user', content: input }]);
    setInput('');
    setIsTyping(true); // Show typing indicator

    try {
      const reply = await askAssistant(input);

      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: reply.answer }
      ]);
    } catch (error) {
      console.error('Error fetching assistant response:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Failed to get response from assistant. Please try again.'
        }
      ]);
    } finally {
      setIsTyping(false); // Hide typing indicator
    }
  };

  return (
    <div className="flex h-screen flex-col bg-gray-50 dark:bg-gray-900">
      {/* Chat header */}
      <header className="sticky top-0 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 p-4">
        <div className="flex items-center justify-between max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-white">Chat-BCG</h1>
          <div className="relative">
            <FileUpload isUploading={isUploading} handleFileUpload={handleFileUpload} />
          </div>
        </div>
      </header>

      {/* Chat messages */}
      <main className="flex-1 overflow-y-auto p-4">
        <ChatMessages messages={messages} isTyping={isTyping} />
      </main>

      {/* Input form */}
      <footer className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 p-4">
        <ChatInputForm input={input} setInput={setInput} handleSubmit={handleSubmit} />
      </footer>
    </div>
  );
}
