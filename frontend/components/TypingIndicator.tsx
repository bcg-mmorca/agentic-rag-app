export default function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-lg p-4 bg-white dark:bg-gray-800 text-gray-900 dark:text-white">
        <div className="flex items-center space-x-1">
          <span className="animate-pulse">.</span>
          <span className="animate-pulse delay-200">.</span>
          <span className="animate-pulse delay-400">.</span>
        </div>
      </div>
    </div>
  );
}
