"use client";

import { uploadFiles } from "@/api/segment";
import ImageSlideshowModal from "@/components/ImageSlideshowModal";
import { InferredImage, UploadedFile } from "@/types";
import { Controls, Player } from "@lottiefiles/react-lottie-player";
import Image from "next/image";
import React, {
  useState,
  DragEvent,
  ChangeEvent,
  useRef,
  useEffect,
} from "react";

const FileUploader: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [currentMessageIndex, setCurrentMessageIndex] = useState<number>(0);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [inferImages, setInferImages] = useState<InferredImage[]>([]);

  const filesPerPage = 5;
  const totalPages = Math.ceil(files.length / filesPerPage);

  // Get the files for the current page
  const startIndex = (currentPage - 1) * filesPerPage;
  const endIndex = startIndex + filesPerPage;
  const filesToDisplay = files.slice(startIndex, endIndex);

  // loading message
  const loadingMessages = [
    "Sending your files to the server for Segmentation",
    "Analyzing seismic data for better accuracy",
    "Processing your request, please hold on",
    "Almost there, finalizing the results",
    "Preparing your data for precise segmentation",
    "Reviewing seismic layers for detailed analysis",
    "Optimizing data transfer for maximum efficiency",
    "Loading seismic models, almost ready",
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentMessageIndex((prevIndex) => {
        if (prevIndex === loadingMessages.length - 1) {
          return 1; // Reset to 1 when reaching the end of the array
        }
        return prevIndex + 1;
      });
    }, 4000); // Change message every 4 seconds

    if (!isSubmitting) {
      return clearInterval(interval);
    }

    return () => clearInterval(interval); // Cleanup the interval on component unmount
  }, [isSubmitting, loadingMessages.length]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Helper to generate pagination numbers
  const getPageNumbers = () => {
    const pageNumbers = [];

    if (totalPages <= 7) {
      // Show all page numbers if there are 7 or fewer
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      // Always show the first and last page
      pageNumbers.push(1);
      if (currentPage > 3) pageNumbers.push("...");
      for (
        let i = Math.max(2, currentPage - 1);
        i <= Math.min(totalPages - 1, currentPage + 1);
        i++
      ) {
        pageNumbers.push(i);
      }
      if (currentPage < totalPages - 2) pageNumbers.push("...");
      pageNumbers.push(totalPages);
    }

    return pageNumbers;
  };

  // Generate unique ID for each file
  const generateId = () => Math.random().toString(36).substring(2, 15);

  // Handle drag events
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const uploadedFiles = Array.from(e.dataTransfer.files).map((file) => ({
      file,
      id: generateId(),
      previewUrl: URL.createObjectURL(file),
    }));
    setFiles((prev) => [...prev, ...uploadedFiles]);
  };

  // Handle file input change
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const uploadedFiles = Array.from(e.target.files).map((file) => ({
        file,
        id: generateId(),
        previewUrl: URL.createObjectURL(file),
      }));
      setFiles((prev) => [...prev, ...uploadedFiles]);
    }
  };

  // Remove a file from the list
  const handleRemoveFile = (id: string) => {
    const updatedFiles = files.filter((item) => item.id !== id);
    setFiles(updatedFiles);
  };

  const handleRemoveAll = () => {
    setFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Clear the file input value
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    const { data, message } = await uploadFiles(files);
    if (data) {
      console.log(data);
      setInferImages(data);
      setIsModalOpen(true);
    } else {
      setMessage(message);
    }
    setIsSubmitting(false);
  };

  return (
    <div className="mt-4 w-full">
      <div className="p-6 max-w-lg mx-auto bg-gray-100 rounded-2xl shadow-lg">
        {isSubmitting ? (
          <div className="text-center text-[#523275]">
            <Player
              autoplay
              loop
              src="/loading.json"
              style={{ height: "256px", width: "256px" }}
            >
              <Controls visible={false} />
            </Player>
            <p>{loadingMessages[currentMessageIndex]}</p>
          </div>
        ) : (
          <>
            <div className="mb-2 flex justify-center">
              <span className="text-red-500">{message}</span>
            </div>

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-dashed border-2 ${
                isDragging ? "border-[#8D66CC]" : "border-gray-300"
              } bg-gray-100 rounded-xl p-6 text-center mb-4 transition-all`}
              style={{
                boxShadow: isDragging
                  ? "inset 10px 10px 15px #b8bfc8, inset -5px -5px 10px #ffffff"
                  : "inset 3px 3px 6px #d1d9e6, inset -3px -3px 6px #ffffff",
              }}
            >
              {/* Responsive Ellipsis Pagination Controls */}
              <div className="pagination-bar flex justify-center my-4 space-x-2">
                {getPageNumbers().map((number, index) =>
                  typeof number === "number" ? (
                    <button
                      key={index}
                      onClick={() => handlePageChange(number)}
                      className={`pagination-number px-4 py-2 font-semibold rounded-md transition-all ${
                        currentPage === number
                          ? "pagination-active bg-purple-600 text-white"
                          : "bg-gray-100 text-gray-600"
                      } ${number === 1 ? "pagination-first" : ""} ${
                        number === totalPages ? "pagination-last" : ""
                      }`}
                      style={{
                        boxShadow:
                          currentPage === number
                            ? "inset 2px 2px 5px #6b4ca5, inset -2px -2px 5px #b085f5"
                            : "5px 5px 15px #d1d9e6, -5px -5px 15px #ffffff",
                      }}
                    >
                      {number}
                    </button>
                  ) : (
                    <span key={index} className="px-2 py-2 text-gray-500">
                      {number}
                    </span>
                  )
                )}
              </div>
              {filesToDisplay.map(({ file, id, previewUrl }) => (
                <div
                  key={id}
                  className="flex items-center justify-between p-3 rounded-xl bg-white shadow transition-all mb-2"
                  style={{
                    boxShadow: "5px 5px 15px #d1d9e6, -5px -5px 15px #ffffff",
                  }}
                >
                  <div className="flex items-center space-x-3">
                    {/* Image Preview */}
                    <img
                      src={previewUrl}
                      alt={file.name}
                      className="w-10 h-10 object-cover rounded-md"
                    />
                    <div>
                      <p className="text-sm font-medium">{file.name}</p>
                      <p className="text-xs text-gray-400">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    className="p-2 bg-gray-100 rounded-lg shadow transition-all"
                    style={{
                      boxShadow: "5px 5px 15px #d1d9e6, -5px -5px 15px #ffffff",
                    }}
                    onClick={() => handleRemoveFile(id)}
                  >
                    <svg
                      className="w-4 h-4 text-gray-500"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                  </button>
                </div>
              ))}
              <label htmlFor="file-input">
                {files.length == 0 && (
                  <Image
                    className="mx-auto"
                    src="/add.png"
                    alt="file-upload"
                    width="256"
                    height="256"
                  />
                )}
                <p className="text-gray-500 text-sm">Drag & Drop Files Here</p>
                <p className="text-xs text-gray-400 mt-1">
                  or Click Browse File
                </p>
                <p
                  className="inline-block cursor-pointer mt-2 py-2 px-4 rounded-md transition-all text-white"
                  style={{
                    background: "#523275", // Adjusted blue color similar to the image
                    boxShadow: "2px 2px 2px #8D66CC, -2px -2px 2px #8D66CC", // Neumorphic shadow effect for blue
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "#8D66CC")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = "#523275")
                  }
                >
                  BROWSE FILE
                </p>
              </label>
              <input
                id="file-input"
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileChange}
                className="hidden" // Hides the input element itself
              />
            </div>
            {files.length > 0 && (
              <div className="mt-4 flex justify-between">
                <span className="flex items-center text-red-500 hover:text-red-700 transition-all">
                  <button
                    onClick={handleRemoveAll}
                    className="flex items-center space-x-1"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth="2"
                      stroke="currentColor"
                      className="w-6 h-6"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                    <span>Clear All</span>
                  </button>
                </span>
                <span className="flex items-center text-blue-500 hover:text-blue-700 transition-all">
                  <button
                    onClick={handleSubmit}
                    className="flex items-center space-x-1"
                  >
                    <Image
                      src="/send.png"
                      alt="submit"
                      width="32"
                      height="32"
                    />
                    <span>Submit</span>
                  </button>
                </span>
              </div>
            )}
          </>
        )}
      </div>
      <ImageSlideshowModal
        base64Images={inferImages}
        isModalOpen={isModalOpen}
        setIsModalOpen={setIsModalOpen}
        setInferImages={setInferImages}
      />
    </div>
  );
};

export default FileUploader;
