"""Hit detection from the overhead (bird's-eye) camera: run the overhead YOLO on a frame stream, track
Misty and Cozmo, and declare a *hit* when the two robots come closer than a threshold or their boxes
overlap. Reports the time until the first hit.

    # one recorded session (overhead_images/ + timestamps from labels.csv)
    python -m overhead.hit_detect --weights runs_det/overhead_v1/weights/best.pt \
        --session "../data_download/chaser/장애물 1개/상대방 움직임 없음/20261103T034629_791792Z" --out overhead/results
    # a video file or a directory of frames
    python -m overhead.hit_detect --weights ... --video game.mp4 --fps 10 --out overhead/results
    # every session → overhead/results/index.json (consumed by overhead/ui)
    python -m overhead.hit_detect --weights ... --sessions ../data_download/chaser/*/*/*/ ../data_download/evader/*/*/ --out overhead/results

Hit rule (per frame, both robots detected or held from the last --hold frames):
    (armed once edge_gap_px > --arm_gap_px, so robots touching in the start pose do not count)
    cond = edge_gap_px <= --gap_px   (boxes overlap/touch when gap_px = 0)   OR   centre_dist_px <= --dist_px
A hit is confirmed when cond holds for --hit_frames consecutive frames (like the paper's Hit@k, k consecutive
frames); the hit time is the timestamp of the first frame of that run.
Outputs per stream: <out>/<name>.json (per-frame records + summary) and <out>/<name>.mp4 (annotated, H.264).
"""
import argparse, glob, json, os, subprocess, sys, time
import cv2, numpy as np

NAMES = ['misty', 'cozmo', 'obstacle']
COL = {'misty': (255, 128, 0), 'cozmo': (0, 200, 0), 'obstacle': (0, 165, 255)}   # BGR


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def edge_gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def frames_of(a):
    """yield (name, frame_bgr, t_seconds)"""
    if a.session:
        import pandas as pd
        df = pd.read_csv(os.path.join(a.session, 'labels.csv'))
        tcol = 'overhead_received_unix_s' if 'overhead_received_unix_s' in df else None
        seen = set()
        for i, row in df.iterrows():
            rel = row['overhead_image']
            if not isinstance(rel, str) or rel in seen:
                continue
            seen.add(rel)
            t = float(row[tcol]) if tcol else i / a.fps
            yield rel, cv2.imread(os.path.join(a.session, rel)), t
    elif a.video and os.path.isdir(a.video):
        for i, p in enumerate(sorted(glob.glob(os.path.join(a.video, '*.jpg')) + glob.glob(os.path.join(a.video, '*.png')))):
            yield os.path.basename(p), cv2.imread(p), i / a.fps
    else:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or a.fps
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield f'{i:06d}', fr, i / fps
            i += 1


class Writer:
    """H.264 mp4 through ffmpeg (browser-playable)."""
    def __init__(self, path, w, h, fps):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                                   '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps:.3f}', '-i', '-',
                                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
                                   '-movflags', '+faststart', path], stdin=subprocess.PIPE)

    def write(self, fr):
        self.p.stdin.write(fr.tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def run_stream(model, a, name, out_dir):
    recs, hold = [], {'misty': None, 'cozmo': None}
    run_len, first_hit, hit_frames, armed = 0, None, 0, a.arm_gap_px < 0
    writer, t0, W, H = None, None, None, None
    ta = time.time()
    for fname, fr, t in frames_of(a):
        if fr is None:
            continue
        if t0 is None:
            t0 = t
        H, W = fr.shape[:2]
        res = model.predict(fr, imgsz=a.imgsz, conf=a.conf, verbose=False, device=a.device)[0]
        det = {'obstacle': []}
        for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            n = NAMES[int(c)]
            if n == 'obstacle':
                det['obstacle'].append([*b, s])
            elif n not in det or s > det[n][4]:
                det[n] = [*b, s]
        boxes = {}
        for r in ('misty', 'cozmo'):
            if r in det:
                hold[r] = {'box': det[r][:4], 'conf': det[r][4], 'age': 0}
            elif hold[r] is not None and hold[r]['age'] < a.hold:
                hold[r] = {**hold[r], 'age': hold[r]['age'] + 1}
            else:
                hold[r] = None
            if hold[r] is not None:
                boxes[r] = hold[r]
        rec = {'frame': fname, 't': round(t - t0, 3), 'obstacles': [[round(v, 1) for v in o] for o in det['obstacle']]}
        for r in ('misty', 'cozmo'):
            rec[r] = {'box': [round(v, 1) for v in boxes[r]['box']], 'conf': round(boxes[r]['conf'], 3),
                      'held': boxes[r]['age']} if r in boxes else None
        cond = False
        if 'misty' in boxes and 'cozmo' in boxes:
            bm, bc = boxes['misty']['box'], boxes['cozmo']['box']
            rec['dist'] = round(float(np.hypot(*np.subtract(centre(bm), centre(bc)))), 1)
            rec['gap'] = round(edge_gap(bm, bc), 1)
            rec['iou'] = round(iou(bm, bc), 3)
            armed = armed or rec['gap'] > a.arm_gap_px   # ignore contact in the start pose until they separate
            cond = armed and (rec['gap'] <= a.gap_px or rec['dist'] <= a.dist_px)
        rec['cond'] = cond
        run_len = run_len + 1 if cond else 0
        hit = run_len >= a.hit_frames
        if hit:
            hit_frames += 1
            if first_hit is None:
                first_hit = {'frame_idx': len(recs) - a.hit_frames + 1, 'frame': recs[-(a.hit_frames - 1)]['frame'] if a.hit_frames > 1 else fname,
                             't': recs[-(a.hit_frames - 1)]['t'] if a.hit_frames > 1 else rec['t']}
        rec['hit'] = hit
        recs.append(rec)

        # ── annotate ──
        if not a.no_video:
            if writer is None:
                writer = Writer(os.path.join(out_dir, name + '.mp4'), W, H, a.fps)
            vis = fr.copy()
            for o in det['obstacle']:
                cv2.rectangle(vis, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), COL['obstacle'], 2)
            for r in ('misty', 'cozmo'):
                if r in boxes:
                    b = boxes[r]['box']; col = COL[r] if boxes[r]['age'] == 0 else (160, 160, 160)
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
                    cv2.putText(vis, f'{r} {boxes[r]["conf"]:.2f}' + (' (held)' if boxes[r]['age'] else ''),
                                (int(b[0]), max(12, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if 'dist' in rec:
                cm, cc = centre(boxes['misty']['box']), centre(boxes['cozmo']['box'])
                cv2.line(vis, (int(cm[0]), int(cm[1])), (int(cc[0]), int(cc[1])), (0, 0, 255) if cond else (200, 200, 0), 2)
                cv2.putText(vis, f'dist {rec["dist"]:.0f}px  gap {rec["gap"]:.0f}px', (10, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(vis, f'dist {rec["dist"]:.0f}px  gap {rec["gap"]:.0f}px', (10, H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            txt = f't={rec["t"]:6.2f}s  #{len(recs)-1}' + (f'  first hit @ {first_hit["t"]:.2f}s' if first_hit else '')
            cv2.putText(vis, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if hit:
                cv2.rectangle(vis, (0, 0), (W - 1, H - 1), (0, 0, 255), 12)
                cv2.putText(vis, 'HIT', (W // 2 - 60, 70), cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 5)
            writer.write(vis)
    if writer is not None:
        writer.close()
    n = len(recs)
    summary = {'name': name, 'frames': n, 'duration_s': recs[-1]['t'] if recs else 0,
               'first_hit': first_hit, 'time_to_first_hit_s': first_hit['t'] if first_hit else None,
               'hit_frames': hit_frames, 'cond_frames': int(sum(r['cond'] for r in recs)),
               'misty_detected': int(sum(r['misty'] is not None and r['misty']['held'] == 0 for r in recs)),
               'cozmo_detected': int(sum(r['cozmo'] is not None and r['cozmo']['held'] == 0 for r in recs)),
               'min_dist': min([r['dist'] for r in recs if 'dist' in r], default=None),
               'params': {k: getattr(a, k) for k in ('gap_px', 'dist_px', 'arm_gap_px', 'hit_frames', 'hold', 'conf', 'imgsz')},
               'fps_processing': round(n / max(time.time() - ta, 1e-6), 1),
               'video': name + '.mp4' if not a.no_video else None}
    json.dump({'summary': summary, 'frames': recs}, open(os.path.join(out_dir, name + '.json'), 'w'))
    print(f'{name}: {n} frames, first hit: {first_hit}, hit frames {hit_frames}, '
          f'det misty {summary["misty_detected"]}/{n} cozmo {summary["cozmo_detected"]}/{n}  ({summary["fps_processing"]} fps)', flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--weights', required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--session', help='recorded session dir (overhead_images/ + labels.csv timestamps)')
    src.add_argument('--sessions', nargs='+', help='many session dirs → index.json')
    src.add_argument('--video', help='video file or directory of frames')
    ap.add_argument('--out', default='overhead/results')
    ap.add_argument('--fps', type=float, default=10.0, help='frame rate when no timestamps (and for the output video)')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--device', default='0')
    ap.add_argument('--gap_px', type=float, default=0.0, help='hit if box edge gap <= this (0 = boxes touch/overlap)')
    ap.add_argument('--dist_px', type=float, default=0.0, help='hit if centre distance <= this (px; ~620 px/m; 0 = off)')
    ap.add_argument('--arm_gap_px', type=float, default=30.0, help='count hits only after the robots were once more than this apart (-1 = always)')
    ap.add_argument('--hit_frames', type=int, default=3, help='consecutive frames needed to confirm a hit')
    ap.add_argument('--hold', type=int, default=5, help='keep a lost box for this many frames')
    ap.add_argument('--no_video', action='store_true')
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    os.makedirs(a.out, exist_ok=True)
    if a.sessions:
        index = []
        for s in a.sessions:
            a.session = s.rstrip('/')
            parts = a.session.split('/')
            name = '__'.join(parts[-3:]).replace(' ', '_')
            summ = run_stream(model, a, name, a.out)
            summ['session'] = a.session
            summ['robot'] = 'chaser' if '/chaser/' in a.session else 'evader'
            summ['condition'] = parts[-2]
            index.append(summ)
        json.dump(index, open(os.path.join(a.out, 'index.json'), 'w'), indent=1)
        hits = [i['time_to_first_hit_s'] for i in index if i['time_to_first_hit_s'] is not None]
        print(f'\n{len(index)} streams, {len(hits)} with a hit; time to first hit: '
              f'median {np.median(hits) if hits else float("nan"):.1f}s  mean {np.mean(hits) if hits else float("nan"):.1f}s')
    else:
        name = os.path.splitext(os.path.basename((a.session or a.video).rstrip('/')))[0]
        run_stream(model, a, name, a.out)


if __name__ == '__main__':
    main()
