import { useCallback, useRef, useState } from "react";

let audioCtx: AudioContext | null = null;

/** Call synchronously inside a tap handler, or iOS will block the beeps. */
export function primeAudio() {
  audioCtx ??= new AudioContext();
  if (audioCtx.state === "suspended") void audioCtx.resume();
}

function beep(freq: number, ms = 150) {
  navigator.vibrate?.(ms);
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  gain.gain.value = 0.2;
  osc.frequency.value = freq;
  osc.connect(gain).connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + ms / 1000);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function pickMimeType() {
  return ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((t) =>
    MediaRecorder.isTypeSupported(t),
  );
}

export function useRecorder() {
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);

  const start = useCallback(async () => {
    if (recorder.current) return; // already starting
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = pickMimeType();
    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunks.current = [];
    rec.ondataavailable = (e) => {
      if (e.data.size) chunks.current.push(e.data);
    };
    recorder.current = rec;
    beep(880); // high beep = speak now
    await sleep(300); // so the beep itself is not recorded
    if (recorder.current !== rec) return; // cancelled meanwhile
    rec.start();
    setRecording(true);
  }, []);

  const stop = useCallback(
    () =>
      new Promise<Blob>((resolve) => {
        const rec = recorder.current;
        if (!rec || rec.state === "inactive") return resolve(new Blob());
        rec.onstop = () => {
          rec.stream.getTracks().forEach((t) => t.stop());
          recorder.current = null;
          beep(440); // low beep = recording ended
          resolve(new Blob(chunks.current, { type: rec.mimeType }));
        };
        rec.stop();
        setRecording(false);
      }),
    [],
  );

  const cancel = useCallback(() => {
    const rec = recorder.current;
    if (!rec) return;
    recorder.current = null;
    rec.onstop = null;
    if (rec.state !== "inactive") rec.stop();
    rec.stream.getTracks().forEach((t) => t.stop());
    setRecording(false);
  }, []);

  return { recording, start, stop, cancel };
}
