"use client";

import { useState, useCallback } from "react";
import { Sparkles, Loader2 } from "lucide-react";
import ScenePanel from "@/components/ScenePanel";
import ActorPanel from "@/components/ActorPanel";
import ResultPreview from "@/components/ResultPreview";
import { detectFaces, swapFace, type DetectedFace } from "@/lib/api";

export default function Home() {
  const [sceneFile, setSceneFile] = useState<File | null>(null);
  const [sceneUrl, setSceneUrl] = useState<string | null>(null);
  const [faces, setFaces] = useState<DetectedFace[]>([]);
  const [imageSize, setImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [selectedFaceIndex, setSelectedFaceIndex] = useState<number | null>(
    null
  );
  const [detecting, setDetecting] = useState(false);

  const [actorFile, setActorFile] = useState<File | null>(null);
  const [actorUrl, setActorUrl] = useState<string | null>(null);

  const [swapping, setSwapping] = useState(false);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("hyperswap_1c_256");

  const handleSceneUpload = useCallback(async (file: File) => {
    setSceneFile(file);
    setSceneUrl(URL.createObjectURL(file));
    setFaces([]);
    setSelectedFaceIndex(null);
    setResultUrl(null);
    setError(null);

    setDetecting(true);
    try {
      const result = await detectFaces(file);
      setFaces(result.faces);
      setImageSize({
        width: result.image_width,
        height: result.image_height,
      });
      if (result.faces.length === 1) {
        setSelectedFaceIndex(0);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Face detection failed");
    } finally {
      setDetecting(false);
    }
  }, []);

  const handleActorUpload = useCallback((file: File) => {
    setActorFile(file);
    setActorUrl(URL.createObjectURL(file));
    setResultUrl(null);
    setError(null);
  }, []);

  const handleActorRemove = useCallback(() => {
    setActorFile(null);
    setActorUrl(null);
    setResultUrl(null);
  }, []);

  const handleSwap = useCallback(async () => {
    if (!sceneFile || !actorFile || selectedFaceIndex === null) return;

    setSwapping(true);
    setError(null);
    try {
      const blob = await swapFace(sceneFile, actorFile, selectedFaceIndex, model);
      setResultUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Face swap failed");
    } finally {
      setSwapping(false);
    }
  }, [sceneFile, actorFile, selectedFaceIndex, model]);

  const canSwap =
    sceneFile && actorFile && selectedFaceIndex !== null && !swapping;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center justify-between px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <h1 className="text-[16px] font-semibold tracking-tight">
            Face Composite
          </h1>
          <span className="text-[12px] font-mono text-text-tertiary">
            &humans casting tool
          </span>
        </div>
      </header>

      <main className="flex-1 flex gap-6 p-6">
        <ScenePanel
          sceneUrl={sceneUrl}
          faces={faces}
          selectedFaceIndex={selectedFaceIndex}
          imageSize={imageSize}
          onSelectFace={setSelectedFaceIndex}
          onUpload={handleSceneUpload}
          loading={detecting}
        />
        <ActorPanel
          actorUrl={actorUrl}
          onUpload={handleActorUpload}
          onRemove={handleActorRemove}
        />
      </main>

      <footer className="flex items-center justify-between px-6 py-4 border-t border-border bg-surface-secondary">
        <div className="flex items-center gap-4">
          {error && (
            <span className="text-[13px] text-error max-w-md truncate">
              {error}
            </span>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-mono text-text-tertiary uppercase tracking-wide">
              Model
            </span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-surface-tertiary text-text-primary text-[13px] rounded-lg px-3 py-1.5 border border-border focus:border-accent outline-none"
            >
              <option value="hyperswap_1c_256">HyperSwap 256</option>
              <option value="inswapper_128">InSwapper 128</option>
            </select>
          </div>

          <button
            onClick={handleSwap}
            disabled={!canSwap}
            className="flex items-center gap-2 px-6 py-2.5 rounded-full text-[14px] font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed bg-accent text-white hover:bg-accent-hover"
          >
            {swapping ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Sparkles size={14} />
                Swap Face
              </>
            )}
          </button>
        </div>
      </footer>

      <ResultPreview
        resultUrl={resultUrl}
        sceneUrl={sceneUrl}
        onClose={() => setResultUrl(null)}
      />
    </div>
  );
}
