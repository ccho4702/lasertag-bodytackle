import sys, os, glob, json, time
import numpy as np, pandas as pd
from PIL import Image
from collections import Counter
from deploy import Agent

ckpt, sessions = sys.argv[1], sys.argv[2:]
def bacc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
P, Y, MS = [], [], []
for s in sessions:
    agent = Agent(ckpt)
    df = pd.read_csv(os.path.join(s, 'labels.csv'))
    gt = np.asarray(agent.robot.load_actions(s))
    pr = []
    for i in range(len(df)):
        fr = np.array(Image.open(os.path.join(s, df.image[i])).convert('RGB'))
        t0 = time.perf_counter(); _, raw = agent.step(fr); MS.append((time.perf_counter() - t0) * 1e3)
        pr.append(raw)
    pr = np.sign(np.round(np.asarray(pr))); g = np.sign(gt[:len(pr)])
    P.append(pr); Y.append(g)
    print(f'{os.path.basename(s.rstrip("/"))}: n={len(pr)} move {np.mean(pr[:,0]==g[:,0]):.3f} turn {np.mean(pr[:,1]==g[:,1]):.3f} joint {np.mean((pr==g).all(1)):.3f}', flush=True)
P, Y = np.concatenate(P), np.concatenate(Y)
jp = (P[:,0]*3+P[:,1]).astype(int); jy = (Y[:,0]*3+Y[:,1]).astype(int)
maj = Counter(jy.tolist()).most_common(1)[0]
name = {(a*3+b): f'v{a:+d}/w{b:+d}' for a in (-1,0,1) for b in (-1,0,1)}
print('ALL n=%d' % len(jy))
print(' move acc %.3f bacc %.3f (majority %.3f)' % (np.mean(P[:,0]==Y[:,0]), bacc(Y[:,0],P[:,0]), max(Counter(Y[:,0].tolist()).values())/len(Y)))
print(' turn acc %.3f bacc %.3f (majority %.3f)' % (np.mean(P[:,1]==Y[:,1]), bacc(Y[:,1],P[:,1]), max(Counter(Y[:,1].tolist()).values())/len(Y)))
print(' joint acc %.3f bacc %.3f (majority %.3f = %s)' % (np.mean(jp==jy), bacc(jy,jp), maj[1]/len(jy), name[maj[0]]))
print(' label dist', {name[k]: round(v/len(jy),3) for k,v in sorted(Counter(jy.tolist()).items())})
print(' pred  dist', {name[k]: round(v/len(jp),3) for k,v in sorted(Counter(jp.tolist()).items())})
print(' median step %.0f ms' % np.median(MS[3:]))
