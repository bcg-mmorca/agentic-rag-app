'use client';

import { askAssistant, processFile } from '@/actions/api';
import { useState } from 'react';

export default function Home() {
  const [messages, setMessages] = useState<Array<{role: 'user' | 'assistant', content: string}>>([]);
  const [input, setInput] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      await processFile(file);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `File "${file.name}" is being processed...`
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

    try {
      const reply = await askAssistant(input);

      setMessages(prev => [...prev, { role: 'assistant', content: reply.content }]);
    } catch (error) {
      console.error('Error fetching assistant response:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Failed to get response from assistant. Please try again.'
      }]);
    }

    setInput('');
  };

  return (
    <div className="flex h-screen flex-col bg-gray-50 dark:bg-gray-900">
      {/* Chat header */}
      <header className="sticky top-0 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 p-4">
        <div className="flex items-center justify-between max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-white">BCG-AI Chat</h1>
          <div className="relative">
            <input
              type="file"
              id="file-upload"
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.doc,.docx,.txt"
              disabled={isUploading}
            />
            <label
              htmlFor="file-upload"
              className={`inline-flex items-center px-4 py-2 rounded-lg 
                ${isUploading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600 cursor-pointer'
                } text-white font-medium text-sm`}
            >
              {isUploading ? 'Uploading...' : 'Upload File'}
            </label>
          </div>
        </div>
      </header>

      {/* Chat messages */}
      <main className="flex-1 overflow-y-auto p-4">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
              Start a conversation by typing a message below.
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-white'
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))
          )}
        </div>
      </main>

      {/* Input form */}
      <footer className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-800 p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 p-4 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            className="rounded-lg bg-blue-500 px-6 py-4 font-semibold text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Send
          </button>
        </form>
      </footer>
    </div>
  );
}
