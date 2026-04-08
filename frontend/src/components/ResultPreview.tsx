"use client";

import { useState, useRef, useCallback } from "react";
import { Download, X, GripVertical } from "lucide-react";

interface ResultPreviewProps {
  resultUrl: string | null;
  sceneUrl: string | null;
  onClose: () => void;
}

export default function ResultPreview({
  resultUrl,
  sceneUrl,
  onClose,
}: ResultPreviewProps) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateSlider = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSliderPos(Math.max(0, Math.min(100, pct)));
  }, []);

  const handleDownload = useCallback(() => {
    if (!resultUrl) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `composite-${Date.now()}.png`;
    a.click();
  }, [resultUrl]);

  if (!resultUrl) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-surface-overlay backdrop-blur-sm">
      <div className="bg-surface-secondary border border-border rounded-2xl max-w-[900px] w-full mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <span className="text-[12px] font-medium tracking-[0.3px] font-mono uppercase text-text-tertiary">
            Result
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-accent text-white text-[14px] font-medium hover:bg-accent-hover transition-colors"
            >
              <Download size={14} />
              Download PNG
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-surface-tertiary transition-colors"
            >
              <X size={16} className="text-text-secondary" />
            </button>
          </div>
        </div>

        {/* Before/After slider */}
        <div
          ref={containerRef}
          className="relative select-none cursor-col-resize"
          onMouseDown={(e) => {
            dragging.current = true;
            updateSlider(e.clientX);
          }}
          onMouseMove={(e) => {
            if (dragging.current) updateSlider(e.clientX);
          }}
          onMouseUp={() => {
            dragging.current = false;
          }}
          onMouseLeave={() => {
            dragging.current = false;
          }}
        >
          {/* Result (full) */}
          <img
            src={resultUrl}
            alt="Result"
            className="w-full h-auto block"
            draggable={false}
          />

          {/* Original (clipped) */}
          {sceneUrl && (
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ width: `${sliderPos}%` }}
            >
              <img
                src={sceneUrl}
                alt="Original"
                className="h-full object-cover"
                style={{
                  width: containerRef.current
                    ? `${containerRef.current.offsetWidth}px`
                    : "100%",
                }}
                draggable={false}
              />
            </div>
          )}

          {/* Slider line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/80"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/90 border border-white/50 flex items-center justify-center shadow-lg">
              <GripVertical size={14} className="text-surface" />
            </div>
          </div>

          {/* Labels */}
          <span className="absolute top-3 left-3 text-[10px] font-mono uppercase px-2 py-1 rounded bg-black/50 text-white/80">
            Before
          </span>
          <span className="absolute top-3 right-3 text-[10px] font-mono uppercase px-2 py-1 rounded bg-black/50 text-white/80">
            After
          </span>
        </div>
      </div>
    </div>
  );
}
