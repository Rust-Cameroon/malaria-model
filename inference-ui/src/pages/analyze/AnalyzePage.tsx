import * as React from "react";
import { health, predict, type PredictResponse, getApiBase } from "@/api/client";

const AnalyzePage: React.FC = () => {
  const [file, setFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [serverOk, setServerOk] = React.useState<boolean | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<PredictResponse | null>(null);

  React.useEffect(() => {
    let mounted = true;
    health()
      .then(() => mounted && setServerOk(true))
      .catch(() => mounted && setServerOk(false));
    return () => {
      mounted = false;
    };
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setError(null);
    setResult(null);
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    setError(null);
    setResult(null);
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predict(file);
      setResult(res);
    } catch (e: any) {
      setError(e?.message ?? "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const serverBadge = serverOk == null
    ? "bg-gray-500/50"
    : serverOk
    ? "bg-emerald-600"
    : "bg-red-600";

  const serverLabel = serverOk == null ? "checking..." : serverOk ? "online" : "offline";

  return (
    <div className="w-full max-w-5xl mx-auto px-6 sm:px-8 py-16 space-y-6">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">Malaria Smear Analysis</h1>
          <p className="opacity-80 text-sm mt-1">Upload a blood smear image and get a prediction from the model.</p>
        </div>
        <div className={`text-xs text-white px-2 py-1 rounded ${serverBadge}`} title={`API base: ${getApiBase()}`}>
          API {serverLabel}
        </div>
      </header>

      <div
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        className="rounded-2xl border border-white/10 bg-white/5 p-6"
      >
        <div className="flex flex-col md:flex-row gap-6">
          <div className="flex-1">
            <label className="block text-sm mb-2 opacity-80">Choose image</label>
            <input
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-emerald-600 file:text-white hover:file:bg-emerald-700"
            />
            <p className="text-xs opacity-60 mt-2">Or drag & drop an image onto this panel.</p>
          </div>
          <div className="flex-1">
            {previewUrl ? (
              <img src={previewUrl} alt="preview" className="rounded-lg max-h-80 object-contain border border-white/10 w-full" />
            ) : (
              <div className="h-48 rounded-lg border border-dashed border-white/20 flex items-center justify-center opacity-70">
                <span className="text-sm">No image selected</span>
              </div>
            )}
          </div>
        </div>
        <div className="mt-6">
          <button
            onClick={onAnalyze}
            disabled={!file || loading || serverOk === false}
            className={`px-4 py-2 rounded-md text-white ${!file || loading || serverOk === false ? "bg-gray-500/60" : "bg-emerald-600 hover:bg-emerald-700"}`}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          {serverOk === false && (
            <span className="ml-3 text-xs text-red-300">Server is offline. Start the Rust API at http://localhost:8080</span>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
        <h2 className="text-lg font-semibold mb-3">Result</h2>
        {!result && !error && <p className="text-sm opacity-80">No result yet.</p>}
        {error && <p className="text-sm text-red-300">{error}</p>}
        {result && (
          <div className="space-y-3">
            <div className="text-sm">Predicted class: <span className="font-semibold">{result.class}</span></div>
            <div>
              <div className="text-xs opacity-70 mb-1">Probabilities</div>
              <div className="space-y-2">
                <ProbBar label="Uninfected" value={result.probabilities[0]} color="bg-sky-500" />
                <ProbBar label="Parasitized" value={result.probabilities[1]} color="bg-fuchsia-500" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ProbBar: React.FC<{ label: string; value: number; color?: string }> = ({ label, value, color = "bg-emerald-500" }) => {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div>
      <div className="flex justify-between text-xs opacity-75">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 bg-white/10 rounded">
        <div className={`h-2 ${color} rounded`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
};

export default AnalyzePage;
