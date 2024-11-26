import { InferredImage, UploadedFile } from "@/types";
import ky from "ky";

const API_URL =
  process.env.NODE_ENV === "production"
    ? "https://b32f-52-163-94-151.ngrok-free.app"
    : "http://localhost:8080";

// Define a TypeScript type for the API response
type UploadResponse = {
  data?: InferredImage[];
  message: string;
};

const api = ky.create({
  timeout: 30000, // 30,000 milliseconds = 30 seconds
});

export const uploadFiles = async (
  files: UploadedFile[]
): Promise<UploadResponse> => {
  try {
    // Create a FormData object to append the file
    const formData = new FormData();
    files.forEach((f) => {
      formData.append("files", f.file); // Adjust the key based on your API
    });

    // Use ky to make a POST request to the upload endpoint
    const response = await api.post(`${API_URL}/infer`, {
      body: formData,
    });
    if (response.ok) {
      return response.json();
    } else {
      throw new Error(response.statusText);
    }
  } catch (error) {
    console.error("Error uploading file", error);
    return {
      message: "Error infer files. Please try again",
    };
  }
};
