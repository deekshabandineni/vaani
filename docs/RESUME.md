# Putting Vaani on your resume

Only fill in numbers you've measured yourself (evaluate.py, benchmark.py, the app's timings).
Interviewers will ask how you got them.

## Resume entry (template)

**Vaani: Edge-First AI Pronunciation Coach** | PyTorch, ONNX Runtime, FastAPI, Node.js, Azure AI Speech, Docker, Kubernetes
- Trained a 5.3M-parameter Conformer-CTC phoneme recognizer on LibriSpeech + speechocean762 and built
  GOP-based mispronunciation scoring, reaching **[PCC]** correlation with expert phone ratings and **[F1]** detection F1.
- Quantized the model to INT8 ONNX (21.3 → 6.1 MB) and ran it in-browser via onnxruntime-web for
  offline, **[N] ms** on-device feedback; JS signal-processing port verified against Python by parity tests.
- Built a hybrid backend: FastAPI service fusing local GOP with Azure Pronunciation Assessment, and a
  Node.js/WebSocket gateway with an offline-first sync queue and cross-device session handoff.
- Designed L1-aware adaptive practice (spaced repetition per phoneme, seeded with Telugu/Hindi
  confusion priors); deployed with Docker Compose and Kubernetes manifests, with CI and headless-Chrome E2E tests.

## Be ready to explain
- **Why CTC + forced alignment for GOP?** You know the expected phones, so alignment gives every
  phone a time span; the posterior gap between the expected phone and the best competitor
  measures how "right" it sounded. CTC posteriors are peaky, so we renormalize over non-blank classes.
- **Why both on-device and Azure?** Latency and offline use vs. a strong reference with fluency and
  completeness. The local model also explains *what* you said instead, which Azure doesn't.
- **How do you keep browser and server scores identical?** Features and GOP are written
  framework-free and ported line by line; `test_parity.py` runs the JS under Node and compares.
- **What breaks with several gateway replicas?** In-memory WebSocket hub, so a user's devices on
  different pods don't see each other. Fix: Redis pub/sub (sticky sessions as a stopgap).
- **What did the scheduler bug teach you?** Failed phones were scheduled 30 min out, so drills chased
  prior guesses instead of observed failures. Fixed by making failures due immediately and squaring
  weakness. Caught by an integration test.
- **Limitations:** ARPAbet can't represent retroflex vs. alveolar T/D; GOP struggles with deletions;
  the L1 priors are heuristics, not measured on users.
