"use client";

import { InferredImage } from "@/types";
import React, { useState, useEffect, useRef, MouseEvent, TouchEventHandler } from "react";
import NextImage from "next/image"; // Renamed import to avoid conflicts

type ImageSlideshowModalProps = {
  base64Images: InferredImage[];
  isModalOpen: boolean;
  setIsModalOpen: (val: boolean) => void;
  setInferImages: (val: InferredImage[]) => void;
};

const ImageSlideshowModal: React.FC<ImageSlideshowModalProps> = ({
  base64Images,
  isModalOpen,
  setIsModalOpen,
  setInferImages,
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(5);
  const [zoomStyle, setZoomStyle] = useState<React.CSSProperties>({
    display: "none",
  });
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const maxVisibleThumbnails = 5;

  useEffect(() => {
    if (base64Images.length > 0) {
      setCurrentIndex(0);
    }
    setIsActive(false);

    // Set canvas dimensions based on the image
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (img && canvas) {
      canvas.width = img.width;
      canvas.height = img.height;
    }
  }, [base64Images]);

  useEffect(() => {
    if (isActive && base64Images.length > 0) {
      drawImageOnCanvas();
    }
  }, [currentIndex, isActive, base64Images]);

  const drawImageOnCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas && base64Images.length > 0) {
      const ctx = canvas.getContext("2d");
      const img = new window.Image();
      img.src = `data:image/png;base64,${base64Images[currentIndex].base64}`;

      if (ctx)
        img.onload = () => {
          const ratio = window.devicePixelRatio || 1;
          canvas.width = img.width * ratio;
          canvas.height = img.height * ratio;
          ctx.scale(ratio, ratio);

          ctx.clearRect(0, 0, img.width, img.height);
          ctx.drawImage(img, 0, 0, img.width, img.height);
        };
    }
  };

  // Handle mouse and touch events
  const getEventCoordinates = (event: MouseEvent | TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;

    if ("touches" in event) {
      const touch = event.touches[0];
      return {
        x: (touch.clientX - rect.left) * ratio,
        y: (touch.clientY - rect.top) * ratio,
      };
    } else {
      return {
        x: (event.clientX - rect.left) * ratio,
        y: (event.clientY - rect.top) * ratio,
      };
    }
  };

  const handleMouseDown = (e: MouseEvent | TouchEvent) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) {
      const { x, y } = getEventCoordinates(e);
      ctx.beginPath();
      ctx.moveTo(x, y);
    }
    e.preventDefault();
  };

  const handleTouchDown: TouchEventHandler<HTMLCanvasElement> = (e) =>
    handleMouseDown(e);

  const handleMouseMove = (e: MouseEvent | TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (ctx) {
      const { x, y } = getEventCoordinates(e);
      ctx.lineTo(x, y);
      ctx.strokeStyle = "#FDB924";
      ctx.lineWidth = 25;
      ctx.stroke();
    }
    e.preventDefault();
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const saveCanvasAsImage = () => {
    setIsSaving(true);
    const canvas = canvasRef.current;
    if (canvas) {
      const dataURL = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataURL;
      link.download = "canvas-image.png";
      link.click();
    }
    setTimeout(() => {
      setIsSaving(false); // Reset loading state
    }, 1000); // 1-second delay
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setCurrentIndex(0);
    setInferImages([]);
  };

  const nextImage = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex === base64Images.length - 1 ? 0 : prevIndex + 1
    );
  };

  const prevImage = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex === 0 ? base64Images.length - 1 : prevIndex - 1
    );
  };

  const selectImage = (index: number) => {
    setCurrentIndex(index);
  };

  const handleImageHover = (e: MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const circleDiameter = 150;
    const overflowAllowance = 50;

    const xPercent = (x / rect.width) * 100;
    const yPercent = (y / rect.height) * 100;

    const adjustedLeft = Math.max(
      -overflowAllowance,
      Math.min(
        rect.width - circleDiameter + overflowAllowance,
        x - circleDiameter / 2
      )
    );
    const adjustedTop = Math.max(
      -overflowAllowance,
      Math.min(
        rect.height - circleDiameter + overflowAllowance,
        y - circleDiameter / 2
      )
    );

    setZoomStyle({
      display: "block",
      left: `${adjustedLeft}px`,
      top: `${adjustedTop}px`,
      backgroundPosition: `${xPercent}% ${yPercent}%`,
    });
  };

  const handleImageLeave = () => {
    setZoomStyle({ display: "none" });
  };

  const handleZoomChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setZoomLevel(Number(e.target.value));
  };

  return (
    <div>
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="relative bg-white rounded-2xl p-6 w-full max-w-3xl mx-4 sm:mx-6 md:mx-8 lg:mx-10 shadow-lg">
            <button
              onClick={closeModal}
              className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-red-500 hover:text-red-700 rounded-full shadow-md z-10"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth="2"
                stroke="currentColor"
                className="w-8 h-8"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>

            {/* Toggle Switch */}
            <div className="flex flex-row items-center z-10 mb-2">
              <span className="mr-2">Edit</span>
              <div
                className={`relative w-16 h-8 rounded-full cursor-pointer transition-all duration-300 ${
                  isActive
                    ? "bg-[#A5CC66] shadow-inner shadow-lg"
                    : "bg-gray-300 shadow-lg"
                }`}
                onClick={() => setIsActive(!isActive)}
              >
                <div
                  className={`absolute top-1 left-1 w-6 h-6 bg-white rounded-full transition-transform duration-300 transform ${
                    isActive ? "translate-x-8" : "translate-x-0"
                  } shadow-md`}
                ></div>
              </div>
            </div>

            {isActive ? (
              <div className="flex flex-col justify-center">
                {base64Images.length > 0 ? (
                  <>
                    <canvas
                      ref={canvasRef}
                      className="border rounded-lg"
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      onTouchStart={handleMouseDown}
                      onTouchMove={handleMouseMove}
                      onTouchEnd={handleMouseUp}
                    />
                    <button
                      onClick={saveCanvasAsImage}
                      disabled={isSaving}
                      className={`my-2 px-4 py-2 font-semibold rounded-lg text-white shadow-md transition-colors duration-300 ${
                        isSaving
                          ? "bg-[#8D66CC] cursor-not-allowed opacity-70"
                          : "bg-[#523275] hover:bg-[#8D66CC]"
                      }`}
                    >
                      {isSaving ? "Saving..." : "Save as Image"}
                    </button>
                  </>
                ) : (
                  <p className="text-gray-500">No images available</p>
                )}
              </div>
            ) : (
              <div>
                <div className="flex justify-center">
                  {base64Images.length > 0 ? (
                    <div className="flex flex-col items-center relative">
                      <p>{base64Images[currentIndex].name}</p>
                      <div
                        className="relative"
                        onMouseMove={handleImageHover}
                        onMouseLeave={handleImageLeave}
                      >
                        <img
                          ref={imageRef}
                          src={`data:image/png;base64,${base64Images[currentIndex].base64}`}
                          alt={`Slide ${currentIndex + 1}`}
                          className="w-full h-auto rounded-lg max-h-80 object-contain my-4"
                        />
                        <div
                          className="absolute w-36 h-36 border-4 border-white rounded-full overflow-hidden pointer-events-none"
                          style={{
                            ...zoomStyle,
                            backgroundImage: `url(data:image/png;base64,${base64Images[currentIndex].base64})`,
                            backgroundSize: `${zoomLevel * 100}%`,
                          }}
                        ></div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-500">No images available</p>
                  )}
                </div>
                <div className="mt-4 flex items-center justify-center space-x-4">
                  <button onClick={() => setZoomLevel(zoomLevel - 1)}>
                    <NextImage
                      className="mx-auto"
                      src="/zoom-out.png"
                      alt="zoom-out"
                      width={32}
                      height={32}
                    />
                  </button>

                  <input
                    type="range"
                    min="2"
                    max="11"
                    step="1"
                    value={zoomLevel}
                    onChange={handleZoomChange}
                    className="slider"
                    style={{
                      background: `linear-gradient(to right, #A5CC66 ${
                        (zoomLevel - 1) * 11.11
                      }%, #d3d3d3 ${(zoomLevel - 1) * 11.11}%)`,
                    }}
                  />

                  <button onClick={() => setZoomLevel(zoomLevel + 1)}>
                    <NextImage
                      className="mx-auto"
                      src="/zoom-in.png"
                      alt="zoom-in"
                      width={32}
                      height={32}
                    />
                  </button>
                </div>
              </div>
            )}

            {/* Navigation buttons */}
            <div className="flex justify-between items-center mt-6">
              <button
                onClick={prevImage}
                disabled={currentIndex === 0}
                className={`px-4 py-2 font-semibold rounded-lg transition-shadow ${
                  currentIndex === 0
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-[#523275] text-white hover:bg-[#8D66CC]"
                } shadow-md`}
              >
                Previous
              </button>

              <div className="flex space-x-2 overflow-hidden">
                {base64Images
                  .slice(0, maxVisibleThumbnails)
                  .map((image, index) => (
                    <button
                      key={index}
                      onClick={() => selectImage(index)}
                      className={`w-12 h-12 border-2 ${
                        currentIndex === index
                          ? "border-[#8D66CC]"
                          : "border-gray-300"
                      } rounded-lg overflow-hidden`}
                    >
                      <img
                        src={`data:image/png;base64,${image.base64}`}
                        alt={`Thumbnail ${index + 1}`}
                        className="w-full h-full object-cover"
                      />
                    </button>
                  ))}
                {base64Images.length > maxVisibleThumbnails && (
                  <span className="text-gray-500">...</span>
                )}
              </div>

              <button
                onClick={nextImage}
                disabled={currentIndex === base64Images.length - 1}
                className={`px-4 py-2 font-semibold rounded-lg transition-shadow ${
                  currentIndex === base64Images.length - 1
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-[#523275] text-white hover:bg-[#8D66CC]"
                } shadow-md`}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageSlideshowModal;
