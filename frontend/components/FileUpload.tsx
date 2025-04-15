interface FileUploadProps {
  isUploading: boolean;
  handleFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function FileUpload({ isUploading, handleFileUpload }: FileUploadProps) {
  return (
    <>
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
    </>
  );
}
