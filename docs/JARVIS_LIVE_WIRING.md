# JARVIS HUD — Live Wiring Patch (paste into `JARVIS Cockpit.dc.html`)

The bridge (`jarvis_bridge.py`) already serves everything the HUD needs:

| Endpoint | Returns | Used for |
|---|---|---|
| `GET /state` | the data contract (account, ftmo, position, **alphas**, policy, perf, clock, `net_signal`, `net_signal_basis`, `gaps`) | every live panel |
| `GET /council?chat=<json>` | OMEGA→JUSTICE→JARVIS transcript + a grounded, **progressive** ruling | the IRAC loop / AGENT COMMS |
| `GET /health` | `{ok, model_attached}` | status |

The bridge is **read-only** (GET only; `POST /order` → 405). Drop your `JARVIS Cockpit.dc.html`
+ `support.js` into the repo root, run `uvicorn jarvis_bridge:app --port 8000`, and open
`http://localhost:8000/JARVIS%20Cockpit.dc.html`. Then add the four methods below to the
`Component` class — that's the whole integration.

---

## 1. Poll live state — add `pullLive()` and start it in `componentDidMount`

```js
// in componentDidMount(), after this.di / this.ci:
this.live    = setInterval(()=>this.pullLive(),   1000);   // /state  ~1/s
this.council = setInterval(()=>this.councilLive(),15000);  // /council ~every 15s
```

```js
async pullLive(){
  try{
    const res = await fetch('http://localhost:8000/state', {cache:'no-store'});
    if(!res.ok) return;                       // bridge down -> simulation keeps running
    const d = await res.json();
    this._liveOn = true;
    const [h,m,s] = d.clock.split(':').map(Number);
    this.setState({
      acct:d.account.balance, equity:d.account.equity,
      dayStart:d.account.day_start_equity, episodeStart:d.account.episode_start_equity,
      peak:d.account.peak_equity,
      dailyPct:d.ftmo.daily_loss_limit_pct, maxPct:d.ftmo.max_drawdown_limit_pct,
      dailyTargetPct:d.ftmo.daily_target_pct,
      pos:{ dir:d.position.dir, sym:d.position.symbol, lots:d.position.lots,
            entry:d.position.entry, price:d.position.price, age:d.position.age_min },
      alphas:d.alphas.map(a=>({ n:a.name, s:a.signal, k:a.streak, dir:a.directional!==false })),
      winRate:d.perf.win_rate_pct, trades:d.perf.trades, consec:d.perf.consecutive_losses,
      dayHistory:d.perf.day_history,
      clk:h*3600+m*60+s,
      _live:d.policy,                         // raw live policy for policySnap()
      _netLive:d.net_signal, _netBasis:d.net_signal_basis,   // honest net (gates excluded)
      _gaps:d.gaps,
    });
  }catch(e){ /* offline -> simulation continues */ }
}
```

## 2. Let live data win — guard `dataTick()` and prefer live in `policySnap()`

```js
dataTick(){
  if(this._liveOn) return;                    // <-- ADD as the first line: stop the random walk when live
  /* ...existing simulation body unchanged... */
}
```

```js
// in policySnap(), at the very top:
if(this.state._live){
  const p=this.state._live;
  return {
    action:p.action,
    action_probs:{buy:p.prob_buy, sell:p.prob_sell, hold:p.prob_hold},
    value_estimate:p.value, confidence:Math.round(p.confidence*100),
    regime:p.regime, expected_drawdown_next_pct:p.expected_dd_pct,
    recommended_size_lots:p.recommended_lots, expected_hold_min:18,
  };
}
/* ...existing net-signal-derived fallback unchanged... */
```

## 3. The honest net signal (fixes the hardcoded "15" + the movement gates)

The bot now has **16 directional alphas** (heading to 18, two of which are non-directional
**movement gates** whose `1` means "market is moving", **not** "buy"). `/state` already sends the
correct directional-only `net_signal` and the count to divide by. Use them — never divide by 15:

```js
netSignal(){
  if(this.state._netLive!=null) return this.state._netLive;          // gates already excluded
  const dir=this.state.alphas.filter(a=>a.dir!==false);              // skip gates locally too
  return dir.length ? dir.reduce((s,a)=>s+a.s,0)/dir.length : 0;     // basis = count, not 15
}
```
*(Optional: render alphas where `a.dir===false` in a distinct colour so a gate's `1` never looks like a LONG vote. The constellation already iterates `this.state.alphas`, so it auto-sizes to 16/18 — no "15" anywhere.)*

## 4. The council reasons for real — add `councilLive()` and call it from `newCycle`

This replaces the simulated OMEGA/JUSTICE/JARVIS text with the **server council**: grounded in the
live system, agent-to-agent, chat-history-aware, and **always progressive**. It works even with no
LLM key (deterministic), and uses the LLM when `ANTHROPIC_API_KEY` is set on the bridge host.

```js
async councilLive(){
  if(!this._liveOn) return;
  try{
    const chat = encodeURIComponent(JSON.stringify(
      (this.state.chat||[]).filter(m=>!m.pending).slice(-8)
        .map(m=>({role:m.who==='user'?'user':'jarvis', text:m.text}))));
    const res = await fetch('http://localhost:8000/council?use_llm=auto&chat='+chat,{cache:'no-store'});
    if(!res.ok) return;
    const co = await res.json();
    co.transcript.forEach(t=>this.pushAgent(t.speaker, t.text));    // OMEGA -> JUSTICE -> JARVIS
    const r=co.ruling;
    this.pushAgent('JARVIS','RULING — '+r.conclusion+'  (posture '+r.posture+', p(pass) '+r.p_pass_pct+'%).');
    if(this.state.voice) this.tts('Council ruling, sir: '+r.progressive_next_step);
  }catch(e){}
}
// in newCycle(): if(this._liveOn){ await this.councilLive(); this.setState({cycling:false}); return; }
```

## 5. (Already wired) `ASK JARVIS` chat stays grounded + progressive

`askClaude()` already sends `snapshot()` + the recent chat history to Claude. To keep it grounded
in the system and always forward-looking, append the council's analysis to the snapshot and one line
to the persona:

```js
// in snapshot(), add:
council_analysis: this.state._council_analysis || null,   // optional: cache co.analysis from councilLive()
gaps: this.state._gaps || [],
// in the persona string, append:
"ALWAYS end your reply with the single next improvement toward passing CONSISTENTLY — never 'nothing to do'. Only cite numbers present in the state; if a figure is in `gaps`, say it is not yet measured."
```

---

That's the whole integration: **four methods** (`pullLive`, `councilLive`, the `dataTick` guard,
the `policySnap`/`netSignal` live-preference) + two persona lines. Everything else (psychology,
P(pass), playbook, red-folder, voice, EXPORT BRIEF) recomputes from the live fields automatically.
Nothing in the HUD's look changes — it just runs on the real bot, and JARVIS's council reasons from
the real system, with the chat, toward a consistent pass.
