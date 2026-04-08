"use client";

import { useCallback, useRef } from "react";
import { Upload, UserCircle, X } from "lucide-react";

interface ActorPanelProps {
  actorUrl: string | null;
  onUpload: (file: File) => void;
  onRemove: () => void;
}

export default function ActorPanel({
  actorUrl,
  onUpload,
  onRemove,
}: ActorPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);

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

  return (
    <div className="flex flex-col gap-3 w-[280px] shrink-0">
      <span className="text-[12px] font-medium tracking-[0.3px] font-mono uppercase text-text-tertiary">
        Actor Headshot
      </span>

      <div
        className="relative rounded-2xl border border-border bg-surface-secondary overflow-hidden cursor-pointer aspect-[3/4] flex items-center justify-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => !actorUrl && inputRef.current?.click()}
      >
        {actorUrl ? (
          <>
            <img
              src={actorUrl}
              alt="Actor headshot"
              className="w-full h-full object-cover"
              draggable={false}
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
              className="absolute top-3 right-3 p-2 rounded-lg bg-surface-overlay backdrop-blur-sm border border-border hover:border-border-strong transition-colors"
            >
              <X size={14} className="text-text-secondary" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
              className="absolute bottom-3 right-3 p-2 rounded-lg bg-surface-overlay backdrop-blur-sm border border-border hover:border-border-strong transition-colors"
            >
              <Upload size={14} className="text-text-secondary" />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 text-text-tertiary p-6">
            <UserCircle size={40} strokeWidth={1} />
            <span className="text-[14px] text-center">
              Drop actor headshot
              <br />
              or click to upload
            </span>
          </div>
        )}
      </div>

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
