"use client";

import { useCallback, useRef } from "react";
import { Upload, ImageIcon } from "lucide-react";
import type { DetectedFace } from "@/lib/api";

interface ScenePanelProps {
  sceneUrl: string | null;
  faces: DetectedFace[];
  selectedFaceIndex: number | null;
  imageSize: { width: number; height: number } | null;
  onSelectFace: (index: number) => void;
  onUpload: (file: File) => void;
  loading: boolean;
}

export default function ScenePanel({
  sceneUrl,
  faces,
  selectedFaceIndex,
  imageSize,
  onSelectFace,
  onUpload,
  loading,
}: ScenePanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) onUpload(file);
    },
    [onUpload]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  // Calculate scale between natural image size and displayed size
  const getScale = () => {
    if (!imgRef.current || !imageSize) return { sx: 1, sy: 1 };
    const rect = imgRef.current.getBoundingClientRect();
    return {
      sx: rect.width / imageSize.width,
      sy: rect.height / imageSize.height,
    };
  };

  return (
    <div className="flex flex-col gap-3 flex-1 min-w-0">
      <div className="flex items-center gap-2">
        <span className="text-[12px] font-medium tracking-[0.3px] font-mono uppercase text-text-tertiary">
          Scene
        </span>
        {loading && (
          <span className="text-[12px] text-text-tertiary animate-pulse">
            Detecting faces...
          </span>
        )}
        {faces.length > 0 && (
          <span className="text-[12px] text-text-tertiary">
            {faces.length} face{faces.length !== 1 ? "s" : ""} detected
          </span>
        )}
      </div>

      <div
        className="relative rounded-2xl border border-border bg-surface-secondary overflow-hidden cursor-pointer min-h-[400px] flex items-center justify-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => !sceneUrl && inputRef.current?.click()}
      >
        {sceneUrl ? (
          <>
            <img
              ref={imgRef}
              src={sceneUrl}
              alt="Scene"
              className="w-full h-auto block"
              draggable={false}
            />
            {/* Face bounding boxes */}
            {faces.map((face) => {
              const { sx, sy } = getScale();
              return (
                <button
                  key={face.index}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectFace(face.index);
                  }}
                  className="absolute border-2 rounded-md transition-all hover:border-accent"
                  style={{
                    left: face.bbox.x * sx,
                    top: face.bbox.y * sy,
                    width: face.bbox.width * sx,
                    height: face.bbox.height * sy,
                    borderColor:
                      selectedFaceIndex === face.index
                        ? "var(--color-accent)"
                        : "rgba(255,255,255,0.5)",
                    backgroundColor:
                      selectedFaceIndex === face.index
                        ? "var(--color-accent-subtle)"
                        : "transparent",
                  }}
                >
                  <span
                    className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-mono px-1.5 py-0.5 rounded"
                    style={{
                      backgroundColor:
                        selectedFaceIndex === face.index
                          ? "var(--color-accent)"
                          : "rgba(0,0,0,0.6)",
                      color: "white",
                    }}
                  >
                    {face.index + 1}
                  </span>
                </button>
              );
            })}
            {/* Replace button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
              className="absolute top-3 right-3 p-2 rounded-lg bg-surface-overlay backdrop-blur-sm border border-border hover:border-border-strong transition-colors"
            >
              <Upload size={14} className="text-text-secondary" />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 text-text-tertiary p-8">
            <ImageIcon size={40} strokeWidth={1} />
            <span className="text-[14px]">
              Drop a scene image or click to upload
            </span>
          </div>
        )}
      </div>

      {faces.length > 0 && selectedFaceIndex === null && (
        <p className="text-[12px] text-warning">
          Click a face in the scene to select the swap target
        </p>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />
    </div>
  );
}
