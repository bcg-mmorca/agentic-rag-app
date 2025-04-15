interface ChatInputFormProps {
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
}

export default function ChatInputForm({
  input,
  setInput,
  handleSubmit,
}: ChatInputFormProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent default Enter behavior
      handleSubmit(e); // Submit the form
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-4">
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown} // Handle keydown events
        placeholder="Type your message... (Shift+Enter for new line)"
        className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 p-4 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        rows={1} // Default height
      />
      <button
        type="submit"
        className="rounded-lg bg-blue-500 px-6 py-4 font-semibold text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        Send
      </button>
    </form>
  );
}
