/* Page logic for the Fashion Frontier demo.
 *
 * Consumes two data files:
 *   data/results.json   generated from benchmarks/results.csv  (do not hand-edit)
 *   data/playbook.json  curated prose                          (edit by hand)
 *
 * Inference lives in assets/inference.js; this file only drives it and draws.
 */
window.FF = window.FF || {};

(function (FF) {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    results: null,
    playbook: null,
    modelId: null,
    sampleIndex: 0,
    truth: null,
    image: null,
    filter: "all",
    scenario: 0,
    busy: false,
  };

  /* ------------------------------------------------------------ helpers */

  /* The repo groups thousands with a space - README, benchmarks/README and
   * playbook.json all print "154 096" and "10 000" - so the demo does too, and
   * the page's prose and its generated figures never disagree. U+00A0 rather
   * than a plain space: the hero prints this inside a sentence, and 154 096
   * must not break across two lines. */
  var THOUSANDS = "\u00A0";
  function fmtInt(n) {
    return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, THOUSANDS);
  }

  function fmtBytes(b) {
    if (b >= 1048576) return (b / 1048576).toFixed(1) + " MB";
    if (b >= 1024) return Math.round(b / 1024) + " KB";
    return b + " B";
  }

  function fmtThroughput(v) {
    if (v === null || v === undefined) return "—";
    if (v >= 10000) return fmtInt(Math.round(v));
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  function pct(v, digits) {
    if (v === null || v === undefined) return "—";
    return (v * 100).toFixed(digits === undefined ? 2 : digits) + "%";
  }

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }

  function modelById(id) {
    var found = null;
    state.results.demo_models.forEach(function (m) { if (m.id === id) found = m; });
    return found;
  }

  /* ------------------------------------------------------------ hero */

  function renderHero() {
    var s = state.results.summary;
    var stats = [
      {
        k: "Board throughput, slowest to fastest point",
        v: fmtInt(Math.round(s.span_x)) + "×",
        cls: "is-fabric",
      },
      {
        k: "That span, in orders of magnitude",
        v: s.decades.toFixed(1),
        cls: "",
      },
      {
        k: "Accuracy given up across the whole span",
        v: s.accuracy_span_pt.toFixed(1) + " pt",
        cls: "is-ref",
      },
      {
        k: "Points measured on the board",
        v: String(s.board_points),
        cls: "is-cpu",
      },
    ];
    $("heroStats").innerHTML = stats.map(function (st) {
      return '<div class="stat ' + st.cls + '"><div class="v">' + esc(st.v) +
        '</div><div class="k">' + esc(st.k) + "</div></div>";
    }).join("");
  }

  /* ------------------------------------------------------------ pickers */

  function renderModels() {
    var box = $("modelPicker");
    box.innerHTML = "";
    state.results.demo_models.forEach(function (m) {
      var b = el("button", "model is-" + m.family);
      b.type = "button";
      b.setAttribute("aria-pressed", String(m.id === state.modelId));
      b.innerHTML =
        '<span class="swatch"></span>' +
        '<span class="model-body">' +
          '<span class="model-name">' + esc(m.id) + "</span>" +
          '<span class="model-meta"><span>' + esc(m.tier) + "</span>" +
          "<span>" + fmtBytes(m.bytes) + "</span>" +
          "<span>" + fmtInt(m.params) + " params</span></span>" +
        "</span>" +
        '<span class="model-acc num">' + pct(m.paper_accuracy) + "</span>";
      b.addEventListener("click", function () {
        if (state.busy) return;
        state.modelId = m.id;
        renderModels();
      });
      box.appendChild(b);
    });
  }

  function renderSamples() {
    var box = $("samplePicker");
    box.innerHTML = "";
    state.results.samples.forEach(function (s, i) {
      var b = el("button", "sample");
      b.type = "button";
      b.title = "Sample " + i + " · true class: " + s.label;
      b.setAttribute("aria-pressed", String(i === state.sampleIndex && state.truth !== null));
      b.innerHTML = '<img alt="' + esc(s.label) + '" src="' + esc(s.file) + '">';
      b.addEventListener("click", function () { if (!state.busy) selectSample(i); });
      box.appendChild(b);
    });
  }

  function selectSample(i) {
    var s = state.results.samples[i];
    if (!s) return;
    state.sampleIndex = i;
    state.truth = s.label_index;
    var img = new Image();
    img.onload = function () {
      state.image = img;
      renderSamples();
      resetOutput();
    };
    img.src = s.file;
  }

  function resetOutput() {
    $("compareBlock").hidden = true;
    $("compareBlock").innerHTML = "";

    var m = modelById(state.modelId);
    var truth = state.truth;
    $("output").innerHTML =
      '<div class="out-head"><h3 class="mono">' + esc(m.id) + "</h3>" +
        (truth === null ? "" :
          '<span class="out-verdict">truth: ' + esc(state.results.classes[truth]) + "</span>") +
      "</div>" +
      '<p class="out-note">' + esc(m.tier) + " · " + esc(m.precision) + " · " +
        fmtInt(m.params) + " parameters · " + fmtBytes(m.bytes) + "</p>" +
      '<p class="out-blurb">' + esc(m.blurb) + "</p>" +
      '<div class="hints">' +
        "<div><b>Predict</b> runs this checkpoint on the selected image and reports the full " +
        "ten-class distribution plus a latency measured on your machine.</div>" +
        "<div><b>Compare all five</b> runs every shipped checkpoint over the same image so the " +
        "latencies sit side by side. The teacher is around forty times slower than the speed tier, " +
        "and that is the whole point.</div>" +
        "<div>Paper accuracy for this checkpoint is <b>" + pct(m.paper_accuracy) + "</b>" +
        (m.board_throughput_img_s
          ? ", measured on the board at <b>" + fmtThroughput(m.board_throughput_img_s) + " img/s</b>."
          : "; the paper records no board throughput for it.") + "</div>" +
      "</div>";
    $("status").textContent = "";
    $("status").className = "status";
  }

  /* ------------------------------------------------------------ output */

  function probabilityBars(ranked, truth) {
    return '<div class="bars">' + ranked.map(function (r, i) {
      var cls = "bar-row" + (i === 0 ? " is-top" : "") + (r.index === truth ? " is-truth" : "");
      return '<div class="' + cls + '">' +
        '<span class="lbl">' + esc(r.label) + "</span>" +
        '<span class="track"><span class="fill" style="width:' +
          Math.max(r.probability * 100, 0.6).toFixed(2) + '%"></span></span>' +
        '<span class="pct">' + (r.probability * 100).toFixed(1) + "%</span>" +
        "</div>";
    }).join("") + "</div>";
  }

  function verdict(predicted, truth) {
    if (truth === null || truth === undefined) {
      return '<span class="out-verdict">no ground truth</span>';
    }
    if (predicted === truth) return '<span class="out-verdict ok">correct</span>';
    return '<span class="out-verdict no">wrong</span>';
  }

  function renderSingle(r) {
    $("compareBlock").hidden = true;
    if (r.error) {
      $("output").innerHTML = '<p class="placeholder">Could not run this model: ' + esc(r.error) + "</p>";
      return;
    }
    var m = r.model;
    var board = m.board_throughput_img_s;
    var mine = r.latencyMs;
    var perImg = 1000 / mine;

    $("output").innerHTML =
      '<div class="out-head"><h3>' + esc(m.predictedLabel || r.predictedLabel) + "</h3>" +
        verdict(r.predicted, state.truth) + "</div>" +
      '<p class="out-note">' +
        "confidence " + pct(r.confidence, 1) +
        (state.truth !== null
          ? " · true class is <strong>" + esc(state.results.classes[state.truth]) + "</strong>"
          : "") +
      "</p>" +
      probabilityBars(r.ranked, state.truth) +
      '<div class="latency">' +
        '<div><div class="k">Median latency</div><div class="v">' + mine.toFixed(1) + " ms</div></div>" +
        '<div><div class="k">Fastest of ' + r.timedRuns + "</div><div class=\"v\">" +
          r.fastestMs.toFixed(1) + " ms</div></div>" +
        '<div><div class="k">Single-image rate</div><div class="v">' +
          (perImg >= 10 ? perImg.toFixed(0) : perImg.toFixed(1)) + " img/s</div></div>" +
        '<div><div class="k">Batch on PYNQ-Z1</div><div class="v">' +
          (board ? fmtThroughput(board) + " img/s" : "—") + "</div></div>" +
        '<div><div class="k">Model load</div><div class="v">' + Math.round(r.loadMs) + " ms</div></div>" +
      "</div>" +
      '<p class="out-note" style="margin-top:16px">' +
        "Measured in this browser with onnxruntime-web " + FF.inference.version +
        " (WASM, one thread): " + r.timedRuns + " timed runs after 3 warm-ups, median reported" +
        (r.batch > 1
          ? ", each run repeating the inference " + r.batch + "× back to back and dividing, " +
            "because the browser clock is coarser than a sub-millisecond model"
          : "") +
        ". The PYNQ column is batch-256 throughput on a dual-core Cortex-A9. " +
        "They are not directly comparable, and that is the interesting part.</p>";
  }

  function renderComparison(results) {
    var truth = state.truth;

    /* Median-of-12 latency, but the compact rows land at 0.3-0.5 ms where the
     * browser timer's own resolution dominates: they tie, and they swap order
     * between runs. So the table makes no claim about a winner. What does repeat
     * is how far the slowest row sits from the fastest, so that is the headline. */
    var minMs = null, maxMs = null, slowest = null;
    results.forEach(function (r) {
      if (r.error) return;
      if (minMs === null || r.latencyMs < minMs) minMs = r.latencyMs;
      if (maxMs === null || r.latencyMs > maxMs) { maxMs = r.latencyMs; slowest = r; }
    });
    var spread = minMs !== null && maxMs !== null && minMs > 0 ? maxMs / minMs : null;

    var rows = results.map(function (r) {
      var m = r.model;
      if (r.error) {
        return "<tr><td class=\"mono\">" + esc(m.id) + "</td><td colspan=\"8\" class=\"pend\">" +
          esc(r.error) + "</td></tr>";
      }
      var v = truth === null ? '<span class="pend">—</span>'
        : (r.predicted === truth ? '<span class="ok">correct</span>' : '<span class="no">wrong</span>');
      return "<tr" + (m.id === state.modelId ? ' class="sel"' : "") + ">" +
        '<td class="mono">' + esc(m.id) + "</td>" +
        "<td>" + pct(m.paper_accuracy) + "</td>" +
        '<td class="dim2">' + fmtBytes(m.bytes) + "</td>" +
        '<td class="dim2">' + fmtInt(m.params) + "</td>" +
        '<td class="mono">' + esc(r.predictedLabel) + "</td>" +
        "<td>" + pct(r.confidence, 1) + "</td>" +
        "<td>" + v + "</td>" +
        "<td><strong>" + r.latencyMs.toFixed(1) + "</strong> ms</td>" +
        "<td>" + fmtThroughput(m.board_throughput_img_s) + "</td>" +
        "</tr>";
    }).join("");

    $("compareBlock").hidden = false;
    $("compareBlock").innerHTML =
      '<div class="out-head"><h3>All five checkpoints, one image</h3>' +
        (spread && spread >= 1.5 && slowest
          ? '<span class="out-verdict ok">' + esc(slowest.model.id) + " took " +
            spread.toFixed(0) + "× the fastest row</span>"
          : "") +
      "</div>" +
      '<p class="out-note">Same input image, same preprocessing, same runtime, same machine. ' +
        "The only thing changing between rows is the network.</p>" +
      '<div class="cmp-scroll"><table class="cmp"><thead><tr>' +
        "<th>Model</th><th>Paper acc</th><th>Size</th><th>Params</th>" +
        "<th>Predicted</th><th>Conf</th><th>Result</th><th>Latency</th><th>PYNQ batch</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table></div>" +
      '<p class="out-note" style="margin-top:16px">' +
        "Note the shape of the last two columns. The 94.77% teacher is the slowest thing here by a " +
        "wide margin, while the compact INT8 tiers are the fastest - which is the entire argument " +
        "of this project expressed in one table. Below a millisecond the browser timer's own " +
        "resolution dominates, so read those rows as a group rather than a ranking.</p>";
    $("compareBlock").scrollIntoView({ block: "nearest" });
  }

  /* ------------------------------------------------------------ actions */

  function setBusy(on, msg) {
    state.busy = on;
    $("runOne").disabled = on;
    $("runAll").disabled = on;
    $("status").className = "status" + (on ? "" : "");
    if (msg !== undefined) $("status").textContent = msg;
  }

  function fail(err) {
    $("status").className = "status err";
    $("status").textContent = err.message || String(err);
  }

  function predictOne() {
    if (state.busy || !state.image) return;
    var m = modelById(state.modelId);
    setBusy(true, m.lazy ? "Downloading " + fmtBytes(m.bytes) + " checkpoint…" : "Running…");
    $("output").innerHTML = '<p class="placeholder">Running ' + esc(m.id) + "…</p>";
    FF.inference.run(state.image, m, state.results.classes).then(function (r) {
      renderSingle(r);
      setBusy(false, "Done.");
    }).catch(function (err) {
      $("output").innerHTML = '<p class="placeholder">Inference failed.</p>';
      setBusy(false, "");
      fail(err);
    });
  }

  function predictAll() {
    if (state.busy || !state.image) return;
    setBusy(true, "Running all five… this downloads about 18 MB the first time.");
    $("output").innerHTML = '<p class="placeholder">Running every checkpoint sequentially…</p>';
    var models = state.results.demo_models;
    FF.inference.runAll(state.image, models, state.results.classes, function (i, m) {
      $("status").textContent = "Running " + (i + 1) + "/" + models.length + " · " + m.id +
        (m.lazy ? " (" + fmtBytes(m.bytes) + " download)" : "");
    }).then(function (results) {
      renderComparison(results);
      setBusy(false, "Done. " + results.length + " checkpoints executed.");
    }).catch(function (err) {
      $("output").innerHTML = '<p class="placeholder">Comparison failed.</p>';
      setBusy(false, "");
      fail(err);
    });
  }

  /* ------------------------------------------------------------ chart */

  /* results.csv ids are precise but unreadable on an axis; these are the short
   * forms the chart and the README figures use. */
  var CHART_NAME = {
    efficientnet_b0_on_arm_cpu: "EfficientNet-B0",
    paper_bnnpynq_lfc_1w1a: "BNN-PYNQ LFC 1W1A",
    tinyplus_kd_fulltrain: "tinyplus_kd",
    tinyfast_xxs: "tinyfast_xxs",
    tinyfast_xxxs: "tinyfast_xxxs",
    tinyfast_xs: "tinyfast_xs",
  };

  function chartName(id) { return CHART_NAME[id] || id; }

  function renderFrontier() {
    var pts = state.results.frontier;
    var tputOnly = state.results.throughput_only;
    var W = 1000, H = 470;
    var ML = 66, MR = 108, MT = 26, MB = 64;
    var X0 = Math.log10(8), X1 = Math.log10(240000);
    var Y0 = 81.5, Y1 = 96.5;

    var px = function (v) { return ML + (Math.log10(v) - X0) / (X1 - X0) * (W - ML - MR); };
    var py = function (a) { return H - MB - (a - Y0) / (Y1 - Y0) * (H - MT - MB); };

    var S = [];
    var key = function (f) {
      return f === "cpu" ? "var(--cpu)" : f === "fabric" ? "var(--fabric)" : "var(--ref)";
    };

    S.push('<rect x="' + ML + '" y="' + MT + '" width="' + (W - ML - MR) + '" height="' +
      (H - MT - MB) + '" fill="none" stroke="var(--line)"/>');

    for (var a = 82; a <= 96; a += 2) {
      S.push('<line x1="' + ML + '" y1="' + py(a) + '" x2="' + (W - MR) + '" y2="' + py(a) +
        '" stroke="var(--line)" stroke-width="1"/>');
      S.push('<text x="' + (ML - 11) + '" y="' + (py(a) + 4) + '" text-anchor="end" font-size="12" ' +
        'fill="var(--muted)">' + a + "%</text>");
    }
    [10, 100, 1000, 10000, 100000].forEach(function (v) {
      S.push('<line x1="' + px(v) + '" y1="' + MT + '" x2="' + px(v) + '" y2="' + (H - MB) +
        '" stroke="var(--line)" stroke-width="1"/>');
      S.push('<text x="' + px(v) + '" y="' + (H - MB + 21) + '" text-anchor="middle" font-size="12" ' +
        'fill="var(--muted)">' + fmtInt(v) + "</text>");
    });

    S.push('<text x="' + ((ML + W - MR) / 2) + '" y="' + (H - 16) + '" text-anchor="middle" ' +
      'font-size="12.5" font-weight="600" fill="var(--muted)">Board throughput (img/s, log scale)</text>');
    S.push('<text x="18" y="' + ((MT + H - MB) / 2) + '" text-anchor="middle" font-size="12.5" ' +
      'font-weight="600" fill="var(--muted)" transform="rotate(-90 18 ' +
      ((MT + H - MB) / 2) + ')">Test accuracy</text>');

    /* the two joins */
    var ref = pts.filter(function (p) { return p.family === "reference"; })[0];
    var cpu = pts.filter(function (p) { return p.family === "cpu"; });
    var fab = pts.filter(function (p) { return p.family === "fabric"; })[0];

    if (ref && cpu.length) {
      S.push('<path d="M' + px(ref.throughput) + " " + py(ref.accuracy) + " L" +
        px(cpu[0].throughput) + " " + py(cpu[0].accuracy) + '" fill="none" stroke="var(--ref)" ' +
        'stroke-width="1.8" stroke-dasharray="6 5"/>');
    }
    if (cpu.length) {
      S.push('<polyline fill="none" stroke="var(--cpu)" stroke-width="2.4" stroke-linejoin="round" ' +
        'points="' + cpu.map(function (p) {
          return px(p.throughput).toFixed(1) + "," + py(p.accuracy).toFixed(1);
        }).join(" ") + '"/>');
    }
    if (cpu.length && fab) {
      var last = cpu[cpu.length - 1];
      S.push('<path d="M' + px(last.throughput) + " " + py(last.accuracy) + " L" +
        px(fab.throughput) + " " + py(fab.accuracy) + '" fill="none" stroke="var(--fabric)" ' +
        'stroke-width="2.2" stroke-dasharray="8 6"/>');
      /* The dashed line is steeper than this label is wide, so anything placed
       * at the midpoint of the line gets crossed by it. Compute where the line
       * sits above the label's right-hand end and drop the text clear of that. */
      var frac = 0.52;
      var ax = px(last.throughput) + (px(fab.throughput) - px(last.throughput)) * frac;
      var ay = py(last.accuracy) + (py(fab.accuracy) - py(last.accuracy)) * frac;
      var rightX = ax + 160;
      var lineAtRight = py(last.accuracy) + (py(fab.accuracy) - py(last.accuracy)) *
        ((rightX - px(last.throughput)) / (px(fab.throughput) - px(last.throughput)));
      ay = Math.max(ay + 26, lineAtRight + 18);
      var leap = fab.throughput / last.throughput;
      S.push('<text x="' + ax.toFixed(0) + '" y="' + ay.toFixed(0) + '" ' +
        'text-anchor="middle" font-size="12.5" font-weight="650" fill="var(--fabric)">' +
        fmtInt(Math.round(leap)) + "× the throughput for " +
        (last.accuracy - fab.accuracy).toFixed(1) + " accuracy points</text>");
    }

    /* throughput-only marker */
    tputOnly.forEach(function (p) {
      var x = px(p.throughput);
      S.push('<line x1="' + x + '" y1="' + (H - MB) + '" x2="' + x + '" y2="330" ' +
        'stroke="var(--faint)" stroke-width="1.4" stroke-dasharray="4 4"/>');
      S.push('<text x="' + (x + 7) + '" y="326" font-size="11.5" font-style="italic" ' +
        'fill="var(--muted)">' + esc(chartName(p.label)) + " · throughput only</text>");
    });

    /* markers */
    var labelled = [];
    pts.forEach(function (p) {
      var x = px(p.throughput), y = py(p.accuracy);
      var c = key(p.family);
      var r = p.family === "fabric" ? 7.5 : 6.5;
      S.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + (r + 9) +
        '" fill="' + c + '" opacity="0.10"/>');
      S.push('<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r +
        '" fill="' + c + '" stroke="var(--card)" stroke-width="2"><title>' +
        esc(chartName(p.label)) + " · " + p.accuracy.toFixed(2) + "% · " +
        fmtInt(Math.round(p.throughput)) + " img/s · " + p.precision + " · " + p.source +
        "</title></circle>");
      labelled.push({ p: p, x: x, y: y, c: c });
    });

    var byFamily = function (f) {
      return labelled.filter(function (l) { return l.p.family === f; });
    };

    /* Reference point sits alone at the far left. Its label goes above the
     * marker, because the dashed connector to the CPU tiers leaves at a shallow
     * angle and would run straight through a right-hand label. */
    byFamily("reference").forEach(function (l, i) {
      S.push('<text x="' + (l.x - 2).toFixed(1) + '" y="' + (l.y - 17 - i * 17) + '" font-size="13.5">' +
        '<tspan font-weight="700" fill="' + l.c + '">' + l.p.accuracy.toFixed(2) + "%</tspan>" +
        '<tspan fill="var(--muted)" font-size="12">  ' + esc(chartName(l.p.label)) + " · " +
        l.p.precision + " · ~" + fmtThroughput(l.p.throughput) + " img/s</tspan></text>");
    });

    /* The four CPU tiers land within about 70 px of each other, because the axis
     * spans four decades. Push the labels apart, then elbow each one back to its
     * marker so there is no doubt which value belongs to which point. */
    var cluster = byFamily("cpu").slice().sort(function (p, q) { return p.y - q.y; });
    var labX = cluster.length ? cluster[cluster.length - 1].x + 58 : 0;
    var lastY = -Infinity;
    var placed = cluster.map(function (l) {
      var ty = Math.max(l.y, lastY + 24);
      lastY = ty;
      return { l: l, ty: ty };
    });
    if (placed.length) {
      var over = placed[placed.length - 1].ty - (H - MB - 8);
      if (over > 0) placed.forEach(function (q) { q.ty -= over; });
    }
    placed.forEach(function (q) {
      S.push('<path d="M' + (q.l.x + 8).toFixed(1) + " " + q.l.y.toFixed(1) +
        " L" + (labX - 17).toFixed(1) + " " + q.l.y.toFixed(1) +
        " L" + (labX - 7).toFixed(1) + " " + q.ty.toFixed(1) +
        '" fill="none" stroke="' + q.l.c + '" stroke-width="1" opacity="0.5"/>');
      S.push('<text x="' + labX.toFixed(1) + '" y="' + (q.ty + 4).toFixed(1) + '" font-size="13.5">' +
        '<tspan font-weight="700" fill="' + q.l.c + '">' + q.l.p.accuracy.toFixed(2) + "%</tspan>" +
        '<tspan fill="var(--muted)" font-size="12">  ' + esc(chartName(q.l.p.label)) +
        "</tspan></text>");
    });
    if (placed.length) {
      S.push('<text x="' + labX.toFixed(1) + '" y="' + (placed[0].ty - 17).toFixed(1) +
        '" font-size="12" font-weight="650" fill="var(--cpu)">four shipped CPU tiers, ' +
        "all inside this hairline</text>");
    }

    /* The fabric point: right-aligned against the plot edge so nothing clips,
     * and lifted well clear of the dashed line arriving from the upper left. */
    byFamily("fabric").forEach(function (l) {
      var rx = W - MR - 8;
      S.push('<text x="' + rx + '" y="' + (l.y - 72) + '" text-anchor="end" font-size="17" ' +
        'font-weight="750" fill="' + l.c + '">' + l.p.accuracy.toFixed(2) + "%</text>");
      S.push('<text x="' + rx + '" y="' + (l.y - 52) + '" text-anchor="end" font-size="13.5" ' +
        'font-weight="650" fill="' + l.c + '">' + fmtInt(Math.round(l.p.throughput)) +
        " img/s</text>");
      S.push('<text x="' + rx + '" y="' + (l.y + 34) + '" text-anchor="end" font-size="12" ' +
        'fill="var(--muted)">' + esc(chartName(l.p.label)) + " · " + l.p.precision +
        "</text>");
    });

    $("frontierSvg").setAttribute("viewBox", "0 0 " + W + " " + H);
    $("frontierSvg").innerHTML = S.join("");
  }

  /* ------------------------------------------------------------ results table */

  var ROUTE_LABEL = {
    teacher: "Teacher / baseline", distill: "Distillation", quant: "Quantisation",
    cpu: "CPU INT8", bnn: "BNN", fpga: "FPGA",
  };

  /* Short forms for the table's route chip: eleven columns leave no room for
   * the filter-button wording. */
  var ROUTE_CHIP = {
    teacher: "Teacher", distill: "Distill", quant: "Quant",
    cpu: "CPU INT8", bnn: "BNN", fpga: "FPGA",
  };

  function renderResultsTable() {
    var rows = state.results.table;
    if (state.filter !== "all") {
      rows = rows.filter(function (r) { return r.route === state.filter; });
    }
    var body = rows.map(function (r) {
      var fam = r.route === "cpu" ? "cpu" : (r.route === "bnn" || r.route === "fpga") ? "fabric" : "";
      return "<tr>" +
        '<td><span class="chip ' + fam + '">' + esc(ROUTE_CHIP[r.route] || r.route) + "</span></td>" +
        '<td class="mono">' + esc(r.model) + "</td>" +
        "<td>" + esc(r.precision) + "</td>" +
        "<td>" + (r.params ? fmtInt(r.params) : '<span class="dim">—</span>') + "</td>" +
        "<td>" + (r.val_accuracy ? pct(r.val_accuracy) : '<span class="dim">—</span>') + "</td>" +
        "<td>" + (r.test_accuracy ? pct(r.test_accuracy) : '<span class="dim">—</span>') + "</td>" +
        "<td>" + (r.board_throughput_img_s ? fmtThroughput(r.board_throughput_img_s) : '<span class="dim">—</span>') + "</td>" +
        "<td>" + (r.board_total_time_s ? r.board_total_time_s.toFixed(1) : '<span class="dim">—</span>') + "</td>" +
        "<td>" + esc(r.device) + "</td>" +
        "<td>" + (r.deliverable === "yes" ? '<span class="chip yes">yes</span>' : '<span class="dim">no</span>') + "</td>" +
        '<td class="mono">' + esc(r.source) + "</td>" +
        "</tr>";
    }).join("");

    $("resultsBody").innerHTML = body || '<tr><td colspan="11" class="dim">Nothing in this route.</td></tr>';
    $("resultsCount").textContent = rows.length + " of " + state.results.table.length + " rows";
  }

  /* ------------------------------------------------------------ playbook */

  function renderPlaybook() {
    var scenes = state.playbook.scenarios;
    var list = $("playList");
    list.innerHTML = "";
    scenes.forEach(function (s, i) {
      var b = el("button", "play-item is-" + s.accent);
      b.type = "button";
      b.setAttribute("aria-pressed", String(i === state.scenario));
      b.innerHTML = '<span class="play-badge">' + esc(s.badge) + "</span>" +
        '<span><span class="play-title">' + esc(s.title) + "</span>" +
        '<span class="play-sub">' + esc(s.subtitle) + "</span></span>";
      b.addEventListener("click", function () { state.scenario = i; renderPlaybook(); });
      list.appendChild(b);
    });

    var s = scenes[state.scenario];
    $("playDetail").className = "play-detail is-" + s.accent;
    $("playDetail").innerHTML =
      "<h3>" + esc(s.title) + "</h3>" +
      '<div class="route">' + esc(s.subtitle) + "</div>" +
      '<div class="play-out num">' + esc(s.outcome) + "</div>" +
      "<p><strong>Constraint.</strong> " + esc(s.constraint) + "</p>" +
      "<p><strong>Route.</strong> " + esc(s.route) + " · <code>" + esc(s.model) + "</code></p>" +
      "<ol>" + s.steps.map(function (t) { return "<li>" + esc(t) + "</li>"; }).join("") + "</ol>" +
      '<div class="play-why"><b>Why this and not the alternative.</b> ' + esc(s.why) + "</div>";

    $("treeHead").textContent = state.playbook.decision_tree.root;
    /* Three columns, three cells: condition | pick | numbers. Nesting .nums
     * inside .pick left the third column an empty div and stacked the figures
     * under the model name instead of aligning them in their own column. */
    $("treeBody").innerHTML = state.playbook.decision_tree.branches.map(function (b) {
      return '<div class="tree-row">' +
        '<div class="cond">' + esc(b.condition) + "</div>" +
        '<div class="pick">→ ' + esc(b.pick) + "</div>" +
        '<div class="nums">' + esc(b.numbers) + "</div>" +
        "</div>";
    }).join("");

    $("caveats").innerHTML = "<h4>Read this before quoting any number</h4><ul>" +
      state.playbook.caveats.map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") +
      "</ul>";
  }

  /* ------------------------------------------------------------ routes */

  function renderRoutes() {
    $("routeList").innerHTML = state.playbook.routes.map(function (r) {
      var chain = r.steps.map(function (s, i) {
        var state_ = i < r.closed ? "done" : (i === r.closed ? "open" : "");
        var link = i > 0 ? '<span class="link' + (i <= r.closed ? " done" : "") + '"></span>' : "";
        return link + '<span class="node"><span class="pip ' + state_ + '"></span>' + esc(s) + "</span>";
      }).join("");
      return '<div class="route is-' + r.accent + '">' +
        "<div><h3>" + esc(r.name) + "</h3>" +
        '<div class="det">' + esc(r.detail) + "</div>" +
        '<div class="chain">' + chain + "</div></div>" +
        '<div class="route-nums">' +
          '<div><div class="k">Accuracy</div><div class="v">' + esc(r.accuracy) + "</div>" +
            '<div class="s">' + esc(r.status) + "</div></div>" +
          '<div><div class="k">Throughput</div><div class="v">' + esc(r.throughput) + "</div>" +
            '<div class="s">' + esc(r.throughput_note) + "</div></div>" +
          '<div><div class="k">Cycle</div><div class="v">' + esc(r.cycle) + "</div>" +
            '<div class="s">&nbsp;</div></div>' +
        "</div></div>";
    }).join("");
  }

  /* ------------------------------------------------------------ wiring */

  function wire() {
    $("runOne").addEventListener("click", predictOne);
    $("runAll").addEventListener("click", predictAll);

    var upload = $("upload");
    var file = $("file");
    upload.addEventListener("click", function () { file.click(); });
    file.addEventListener("change", function () {
      var f = file.files && file.files[0];
      if (!f) return;
      var url = URL.createObjectURL(f);
      var img = new Image();
      img.onload = function () {
        state.image = img;
        state.truth = null;
        state.sampleIndex = -1;
        renderSamples();
        resetOutput();
        $("status").textContent =
          "Loaded " + f.name + ". No ground truth for an upload, so correctness is not shown.";
      };
      img.src = url;
    });

    var filters = ["all", "teacher", "distill", "quant", "cpu", "bnn", "fpga"];
    $("filters").innerHTML = filters.map(function (f) {
      return '<button type="button" data-f="' + f + '" class="' + (f === "all" ? "on" : "") + '">' +
        (f === "all" ? "All routes" : (ROUTE_LABEL[f] || f)) + "</button>";
    }).join("");
    $("filters").addEventListener("click", function (e) {
      var b = e.target.closest("button");
      if (!b) return;
      state.filter = b.dataset.f;
      Array.prototype.forEach.call($("filters").children, function (c) {
        c.classList.toggle("on", c === b);
      });
      renderResultsTable();
    });
  }

  /* ------------------------------------------------------------ boot */

  function boot() {
    if (location.protocol === "file:") {
      document.body.innerHTML =
        '<div style="max-width:640px;margin:80px auto;padding:0 24px;font:16px/1.6 system-ui">' +
        "<h1>Serve this page over HTTP</h1>" +
        "<p>The demo fetches its checkpoints and data with <code>fetch()</code>, which browsers " +
        "block on <code>file://</code> URLs. From the repository root:</p>" +
        "<pre><code>python -m http.server 8000</code></pre>" +
        "<p>then open <code>http://localhost:8000/web/</code>.</p>" +
        "<p>Serving from the repository root rather than from <code>web/</code> keeps the links " +
        "into <code>docs/</code> working.</p></div>";
      return;
    }

    Promise.all([
      fetch("data/results.json").then(function (r) { return r.json(); }),
      fetch("data/playbook.json").then(function (r) { return r.json(); }),
    ]).then(function (both) {
      state.results = both[0];
      state.playbook = both[1];

      /* Default to the balanced tier, not the teacher: it is the model this
       * project actually recommends, and it is 184 KB instead of 15 MB. */
      var preferred = modelById("tinyfast_xxs") || state.results.demo_models[0];
      state.modelId = preferred.id;

      renderHero();
      renderModels();
      renderSamples();
      renderFrontier();
      renderResultsTable();
      renderPlaybook();
      renderRoutes();
      wire();
      selectSample(0);

      /* Warm the default checkpoint so the first click is instant. Failures are
       * silent here - the button will surface them with a real message. */
      FF.inference.load(preferred).catch(function () {});
      if (new URLSearchParams(location.search).has("selftest")) FF.selftest();
    }).catch(function (err) {
      document.getElementById("output").innerHTML =
        '<p class="placeholder">Could not load the data files: ' + esc(err.message || err) + "</p>";
    });
  }

  /* ------------------------------------------------------------ selftest
   *
   * Append ?selftest=1 to the URL and every shipped checkpoint is run against
   * every sample image, with a machine-readable transcript written into the
   * DOM. It exists so the demo's central claim - that these are the paper's
   * real weights and they still classify correctly in a browser - can be
   * checked without anyone clicking a button.
   */
  FF.selftest = function () {
    var out = document.createElement("pre");
    out.id = "selftest";
    out.style.cssText = "position:absolute;left:-9999px;top:0;white-space:pre;";
    document.body.appendChild(out);

    var lines = [];
    function log(s) { lines.push(s); out.textContent = lines.join("\n"); }

    function loadImg(src, attempt) {
      attempt = attempt || 0;
      return new Promise(function (res, rej) {
        var i = new Image();
        i.onload = function () { res(i); };
        i.onerror = function () {
          if (attempt < 2) {
            setTimeout(function () { res(loadImg(src, attempt + 1)); }, 150 * (attempt + 1));
          } else {
            rej(new Error("image failed after retries: " + src));
          }
        };
        i.src = src;
      });
    }

    var models = state.results.demo_models;
    var samples = state.results.samples;
    var images = {};
    var hits = 0, tries = 0;
    var started = Date.now();

    log("ort " + FF.inference.version + " models " + models.length + " samples " + samples.length);

    /* Sequential, not Promise.all: ten simultaneous image requests against a
     * single-threaded static server was observed to drop one at random. */
    var warm = Promise.resolve();
    samples.forEach(function (s) {
      warm = warm.then(function () {
        return loadImg(s.file).then(function (im) {
          images[s.file] = im;
          log("image ok " + s.file);
        });
      });
    });

    warm.then(function () {
      log("images loaded " + Object.keys(images).length);
      var chain = Promise.resolve();
      models.forEach(function (m) {
        chain = chain.then(function () {
          var t = Date.now();
          return FF.inference.load(m).then(function () {
            log("load ok " + m.id + " " + (Date.now() - t) + "ms");
          }).catch(function (e) {
            log("load FAIL " + m.id + " " + (e.message || e));
          });
        });
        samples.forEach(function (s, i) {
          chain = chain.then(function () {
            return FF.inference.run(images[s.file], m, state.results.classes, 0).then(function (r) {
              tries += 1;
              var ok = r.predicted === s.label_index;
              if (ok) hits += 1;
              log("run " + m.id + " s" + i + " " + (ok ? "OK  " : "MISS") +
                " truth=" + s.label + " pred=" + r.predictedLabel +
                " conf=" + r.confidence.toFixed(3) + " ms=" + r.latencyMs.toFixed(1));
            }).catch(function (e) {
              tries += 1;
              log("run " + m.id + " s" + i + " ERR " + (e.message || e));
            });
          });
        });
      });
      return chain;
    }).then(function () {
      var secs = ((Date.now() - started) / 1000).toFixed(1);
      log("TOTAL " + hits + "/" + tries + " correct in " + secs + "s");
      log("SELFTEST DONE");
      document.title = "SELFTEST " + hits + "/" + tries;
    }).catch(function (e) {
      log("FATAL " + (e.message || e));
      log("SELFTEST DONE");
      document.title = "SELFTEST ERROR";
    });
  };

  FF.site = { state: state, boot: boot };
  document.addEventListener("DOMContentLoaded", boot);
})(window.FF);
