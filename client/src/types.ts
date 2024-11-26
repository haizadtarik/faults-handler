export type UploadedFile = {
  file: File;
  id: string;
  previewUrl: string;
};

export type InferredImage = {
  base64: string;
  name: string;
};
