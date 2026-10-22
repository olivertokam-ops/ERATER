import cv2
import time
import csv
import threading
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
import torch

class VisionThesisSystem:
    def __init__(self, output_csv="benchmark_metrics.csv"):
        # Inizializzazione Rilevatore Veloce
        print("[INIT] Caricamento YOLOv8...")
        self.detector = YOLO("yolov8n.pt")
        
        # Inizializzazione VLM (Slow Path)
        print("[INIT] Caricamento Vision-Language Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fallback a Moondream se disponibile, altrimenti mock robusto integrato
        self.vlm_ready = False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = "vikhyatk/moondream2"
            revision = "2024-08-26"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.vlm = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                revision=revision,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.vlm.eval()
            self.vlm_ready = True
            print(f"[INIT] VLM caricato con successo su {self.device.upper()}.")
        except Exception as e:
            print(f"[WARN] VLM locale non caricato ({e}). Attivata modalità euristica per test.")

        # Analytics di scena
        self.trajectories = defaultdict(list)
        self.first_seen = {}
        self.latest_detected_labels = []
        
        # Concorrenza Thread-Safe
        self.vlm_busy = False
        self.latest_interpretation = "Premi 'C' per interpretazione semantica guidata dal contesto."
        
        # Telemetria & Logging
        self.output_csv = output_csv
        self._init_csv()

    def _init_csv(self):
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Timestamp", "FPS", "YOLO_Inference_ms", "Active_Objects_Count"])

    def _vlm_worker(self, frame_bgr, detected_summary):
        """Worker asincrono: analizza l'immagine senza bloccare il video."""
        try:
            rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            prompt = (
                f"In this scene, the real-time detector found: {detected_summary}. "
                "Briefly describe what the person is doing and their interaction with the environment."
            )

            if self.vlm_ready:
                image_embeds = self.vlm.encode_image(pil_img)
                response = self.vlm.answer_question(image_embeds, prompt, self.tokenizer)
            else:
                time.sleep(1.0) # Simula latenza di inferenza VLM
                response = f"Analisi mock: Rilevati [{detected_summary}]. Persona posizionata nell'area di lavoro."

            self.latest_interpretation = response
        except Exception as err:
            self.latest_interpretation = f"Errore VLM: {str(err)}"
        finally:
            self.vlm_busy = False

    def trigger_vlm(self, frame):
        if not self.vlm_busy:
            self.vlm_busy = True
            self.latest_interpretation = "Elaborazione semantica in corso..."
            detected_summary = ", ".join(set(self.latest_detected_labels)) or "Nessun oggetto primario"
            t = threading.Thread(target=self._vlm_worker, args=(frame.copy(), detected_summary))
            t.daemon = True
            t.start()

    def process_analytics(self, frame, boxes, track_ids, class_ids, names):
        """Calcola coordinate, permanenza temporale e prossimità spaziale."""
        h, w, _ = frame.shape
        current_time = time.time()
        current_labels = []

        if track_ids is None:
            return frame, current_labels

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls_id)]
            current_labels.append(label)

            # 1. Tracciamento centroide e traiettoria
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.trajectories[track_id].append((cx, cy))
            if len(self.trajectories[track_id]) > 20:
                self.trajectories[track_id].pop(0)

            # Disegna la scia del movimento
            pts = self.trajectories[track_id]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

            # 2. Dwell Time (tempo di stazionamento)
            if track_id not in self.first_seen:
                self.first_seen[track_id] = current_time
            dwell_time = int(current_time - self.first_seen[track_id])

            # 3. Spatial Proximity (Stima della vicinanza tramite area occupata)
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            area_ratio = (box_area / frame_area) * 100

            proximity_alert = "VICINO" if area_ratio > 25 else "NORMALE"
            color = (0, 0, 255) if proximity_alert == "VICINO" else (0, 255, 0)

            # Render HUD sul singolo oggetto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {label} ({dwell_time}s) [{proximity_alert}]", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame, current_labels

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        prev_time = time.time()

        print("[START] Pipeline avviata. Premi 'C' per interrogare il VLM, 'Q' per terminare.")

        with open(self.output_csv, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start_infer = time.time()
                # Tracking persistente con YOLOv8
                results = self.detector.track(frame, persist=True, verbose=False)
                yolo_latency_ms = (time.time() - start_infer) * 1000

                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.id is not None else None
                track_ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else None
                class_ids = r.boxes.cls.cpu().tolist() if r.boxes.id is not None else None

                # Elaborazione analitica
                annotated_frame, self.latest_detected_labels = self.process_analytics(
                    frame, boxes, track_ids, class_ids, r.names
                )

                # Calcolo FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                frame_count += 1

                # Scrittura telemetria (campionata ogni 5 frame per efficienza)
                if frame_count % 5 == 0:
                    active_count = len(track_ids) if track_ids else 0
                    csv_writer.writerow([frame_count, round(curr_time, 2), round(fps, 1), 
                                         round(yolo_latency_ms, 2), active_count])

                # Overlay Informazioni Superiori (Dashboard HUD)
                cv2.rectangle(annotated_frame, (0, 0), (640, 45), (20, 20, 20), -1)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Latenza YOLO: {yolo_latency_ms:.1f}ms", 
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                vlm_status = "[VLM: Elaborazione...]" if self.vlm_busy else "[VLM: Pronto - Premi 'C']"
                cv2.putText(annotated_frame, vlm_status, (400, 28), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

                # Overlay Risultato Semantico Inferiore
                cv2.rectangle(annotated_frame, (0, 445), (640, 480), (10, 10, 10), -1)
                cv2.putText(annotated_frame, self.latest_interpretation[:80], (10, 468), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                cv2.imshow("Sistema di Visione & Analytics Ambientale", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.trigger_vlm(frame)

        cap.release()
        cv2.destroyAllWindows()
        print(f"[FINISH] Sessione terminata. Metriche salvate in: {self.output_csv}")

if __name__ == "__main__":
    system = VisionThesisSystem()
    system.run()
