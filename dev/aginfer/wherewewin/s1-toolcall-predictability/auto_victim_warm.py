"""Clean AUTONOMOUS warm demo: the DAEMON (not manual) warms a fully-evicted
victim via the action-timeline promote. Victim registers prefix + posts
tool_call_start (schedules promote); flood evicts it; ticker events advance the
event-stream clock past the promote's due time -> daemon fires warm_to_hbm; then
the victim resumes and should hit HBM. vs B (no events -> no warm -> recompute)."""
import requests, time, statistics, sys, json as J
B="http://127.0.0.1:30000"; D="http://127.0.0.1:9100"; V=129000
VICT=50000; FLOOD_N=6; FLOOD_LEN=50000; ETA=6.0; N=3
def seq(s,n): return [(s+i)%V for i in range(n)]
def gen(ids,pid,mx=4):
    requests.post(B+"/generate",json={"input_ids":ids,"sampling_params":{"temperature":0,"max_new_tokens":mx,"ignore_eos":True},"program_id":pid},timeout=300).raise_for_status()
def ev(kind,pid,payload=None):
    try: requests.post(D+"/aginfer/event",json={"kind":kind,"session":pid,**(payload or {})},timeout=10)
    except Exception: pass
def reg(pid,ids):
    try: requests.post(D+"/aginfer/session_prefix",json={"program_id":pid,"input_ids":ids},timeout=15)
    except Exception: pass
def ttft(ids,pid):
    body={"input_ids":ids,"sampling_params":{"temperature":0,"max_new_tokens":6,"ignore_eos":True},"program_id":pid,"stream":True}
    t0=time.perf_counter(); tt=None; c=0
    with requests.post(B+"/generate",json=body,timeout=300,stream=True) as r:
        for line in r.iter_lines():
            if not line: continue
            if tt is None: tt=(time.perf_counter()-t0)*1000
            s=line.decode() if isinstance(line,bytes) else line
            if s.startswith("data:"): s=s[5:].strip()
            if s and s!="[DONE]":
                try: c=int((J.loads(s).get("meta_info") or {}).get("cached_tokens") or c)
                except: pass
            if tt and c: break
    return tt,c
def flood(tag):
    for k in range(FLOOD_N): gen(seq((90000+k*4001+tag*131)%V,FLOOD_LEN),f"FL{tag}_{k}",mx=2)
b_t,o_t=[],[]; warmed=0
for t in range(N):
    vid=f"AV{t}"; vids=seq((t*7001+1)%V,VICT)
    # ---- B: no events, flood, resume (recompute)
    gen(vids,vid,mx=4); time.sleep(0.5); flood(2*t)
    tb,cb=ttft(vids,vid)
    # ---- ours: cache, REGISTER + tool_call_start(eta) -> schedules promote; flood; TICK to fire warm; resume
    gen(vids,vid,mx=4); time.sleep(0.5)
    ev("session_arrival",vid); ev("llm_prefill",vid)
    reg(vid,vids)
    ev("tool_call_start",vid,{"tool_name":"bash","tool_args":{"command":f"sleep {int(ETA)}"},"tool_eta_s":ETA})
    flood(2*t+1)                                   # evict the victim (takes a few s)
    # tick the event-stream clock past the promote due (~ETA s) so the daemon fires warm
    for _ in range(14):
        ev("llm_prefill",f"TICK{t}"); time.sleep(0.5)
    time.sleep(1.5)                                # let the warm prefill complete
    to,co=ttft(vids,vid)
    ev("session_end",vid)
    b_t.append(tb); o_t.append(to)
    print(f"trial {t}: B={tb:.0f}ms(cached={cb})  ours={to:.0f}ms(cached={co})  saved={tb-to:.0f}ms")
print(f"\nB mean={statistics.mean(b_t):.0f}ms  ours(daemon-warm) mean={statistics.mean(o_t):.0f}ms")
sv=statistics.mean(b_t)-statistics.mean(o_t)
print(f"=> daemon autonomous warm saves {sv:.0f}ms ({100*sv/statistics.mean(b_t):.0f}%)" if sv>500 else f"=> only {sv:.0f}ms — investigate timing")
