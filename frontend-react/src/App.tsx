import { useEffect, useRef, useState } from "react";
import { askAI, resetSession, transcribe } from "./api";
import { useRecorder } from "./useRecorder";
import "./App.css";

type Step = "capture" | "ask" | "processing" | "answer";

export default function App() {
  const [step, setStep] = useState<Step>("capture");
  const [image, setImage] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [speak, setSpeak] = useState(false);
  const [lang, setLang] = useState("en");

  const fileInput = useRef<HTMLInputElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const { recording, start, stop } = useRecorder();

  // Move focus to the heading whenever the step changes to tell user about the screen
  useEffect(() => {
    heading.current?.focus();
  }, [step]);

  function say(text: string, code: string) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = code;
    speechSynthesis.speak(u);
  }

  function onPhoto(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setImage(f);
    setStep("ask");
    e.target.value = ""; // allows taking the same file name again
  }

  async function toggleRecord() {
    setError("");
    if (!recording) {
      try {
        await start();
      } catch {
        setError("Microphone not available.");
      }
      return;
    }
    const blob = await stop();
    setStep("processing");
    try {
      const t = await transcribe(blob);
      if (!t.text) {
        setError("I could not hear a question. Please try again.");
        setStep(sessionId ? "answer" : "ask");
        return;
      }
      setLang(t.language_code);
      const res = await askAI({ sessionId, image, t });
      setSessionId(res.session_id);
      setAnswer(res.ai_response);
      setStep("answer");
      if (speak) say(res.ai_response, t.language_code);
    } catch {
      setError("Something went wrong. Please try again.");
      setStep(sessionId ? "answer" : "ask");
    }
  }

  function startOver() {
    if (sessionId) resetSession(sessionId);
    speechSynthesis.cancel();
    setSessionId(null);
    setImage(null);
    setAnswer("");
    setError("");
    setStep("capture");
    setTimeout(() => fileInput.current?.click(), 0); // opens camera right away
  }

  const RecordButton = (
    <button className="big" onClick={toggleRecord} aria-pressed={recording}>
      {recording ? "Stop recording" : "Tap to record your question"}
    </button>
  );

  return (
    <main>
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={onPhoto}
        hidden
        aria-hidden="true"
        tabIndex={-1}
      />

      {step === "capture" && (
        <>
          <h1 ref={heading} tabIndex={-1}>
            Freiheit. Your visual assistant
          </h1>
          <button className="big" onClick={() => fileInput.current?.click()}>
            Take a photo
          </button>
        </>
      )}

      {step === "ask" && (
        <>
          <h1 ref={heading} tabIndex={-1}>
            Photo taken. Ask your question.
          </h1>
          {RecordButton}
          <button className="big secondary" onClick={startOver}>
            Retake photo
          </button>
        </>
      )}

      {step === "processing" && (
        <h1 ref={heading} tabIndex={-1}>
          Getting your answer, please wait.
        </h1>
      )}

      {step === "answer" && (
        <>
          <h1 ref={heading} tabIndex={-1}>
            Answer
          </h1>
          <p className="answer" aria-live="polite">
            {answer}
          </p>
          {RecordButton}
          <button className="big secondary" onClick={() => say(answer, lang)}>
            Repeat answer
          </button>
          <button className="big secondary" onClick={startOver}>
            Take a new photo
          </button>
        </>
      )}

      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}

      <label className="toggle">
        <input
          type="checkbox"
          checked={speak}
          onChange={(e) => setSpeak(e.target.checked)}
        />
        Speak answers aloud automatically
      </label>
    </main>
  );
}
