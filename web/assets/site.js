/* Browser UI for checkpoint inference and historical experiment records. */
window.FF = window.FF || {};

(function (FF) {
  "use strict";

  var $ = function (id) {
    return document.getElementById(id);
  };

  var state = {
    results: null,
    modelId: null,
    sampleIndex: 0,
    truth: null,
    image: null,
    filter: "all",
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
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }

  function modelById(id) {
    var found = null;
    state.results.demo_models.forEach(function (m) {
      if (m.id === id) found = m;
    });
    return found;
  }

  /* ------------------------------------------------------------ hero */

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
        '<span class="model-name">' +
        esc(m.id) +
        "</span>" +
        '<span class="model-meta"><span>' +
        esc(m.tier) +
        "</span>" +
        "<span>" +
        fmtBytes(m.bytes) +
        "</span>" +
        "<span>" +
        fmtInt(m.params) +
        " params</span></span>" +
        "</span>" +
        '<span class="model-acc num">' +
        pct(m.artifact_accuracy) +
        "</span>";
      b.addEventListener("click", function () {
        if (state.busy) return;
        state.modelId = m.id;
        renderModels();
        resetOutput();
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
      b.setAttribute(
        "aria-pressed",
        String(i === state.sampleIndex && state.truth !== null),
      );
      b.innerHTML =
        '<img alt="' + esc(s.label) + '" src="' + esc(s.file) + '">';
      b.addEventListener("click", function () {
        if (!state.busy) selectSample(i);
      });
      box.appendChild(b);
    });
  }

  function selectSample(i) {
    var s = state.results.samples[i];
    if (!s) return;
    state.sampleIndex = i;
    state.truth = s.label_index;
    state.image = null;
    var img = new Image();
    img.onload = function () {
      state.image = img;
      renderSamples();
      resetOutput();
    };
    img.onerror = function () {
      fail(new Error("Could not load sample image. Please retry."));
    };
    img.src = s.file;
  }

  function resetOutput() {
    $("compareBlock").hidden = true;
    $("compareBlock").innerHTML = "";

    var m = modelById(state.modelId);
    var truth = state.truth;
    $("output").innerHTML =
      '<div class="out-head"><h3 class="mono">' +
      esc(m.id) +
      "</h3>" +
      (truth === null
        ? ""
        : '<span class="out-verdict">truth: ' +
          esc(state.results.classes[truth]) +
          "</span>") +
      "</div>" +
      '<p class="out-note">' +
      esc(m.tier) +
      " · " +
      esc(m.precision) +
      " · " +
      fmtInt(m.params) +
      " parameters · " +
      fmtBytes(m.bytes) +
      "</p>" +
      '<p class="out-blurb">' +
      esc(m.blurb) +
      "</p>" +
      '<div class="hints">' +
      "<div>Select <b>Predict</b> to view the class scores, or <b>Compare all five</b> to test the same image with each model.</div>" +
      "<div>Checkpoint test accuracy: <b>" +
      pct(m.artifact_accuracy) +
      "</b> on 10,000 images.</div>" +
      "<div>Paper accuracy for this checkpoint is <b>" +
      pct(m.paper_accuracy) +
      "</b>" +
      (m.board_throughput_img_s
        ? ", measured on the board at <b>" +
          fmtThroughput(m.board_throughput_img_s) +
          " img/s</b>."
        : "; the paper records no board throughput for it.") +
      "</div>" +
      "</div>";
    $("status").textContent = "";
    $("status").className = "status";
  }

  /* ------------------------------------------------------------ output */

  function probabilityBars(ranked, truth) {
    return (
      '<div class="bars">' +
      ranked
        .map(function (r, i) {
          var cls =
            "bar-row" +
            (i === 0 ? " is-top" : "") +
            (r.index === truth ? " is-truth" : "");
          return (
            '<div class="' +
            cls +
            '">' +
            '<span class="lbl">' +
            esc(r.label) +
            "</span>" +
            '<span class="track"><span class="fill" style="width:' +
            Math.max(r.probability * 100, 0.6).toFixed(2) +
            '%"></span></span>' +
            '<span class="pct">' +
            (r.probability * 100).toFixed(1) +
            "%</span>" +
            "</div>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function verdict(predicted, truth) {
    if (truth === null || truth === undefined) {
      return '<span class="out-verdict">no ground truth</span>';
    }
    if (predicted === truth)
      return '<span class="out-verdict ok">correct</span>';
    return '<span class="out-verdict no">wrong</span>';
  }

  function renderSingle(r) {
    $("compareBlock").hidden = true;
    if (r.error) {
      $("output").innerHTML =
        '<p class="placeholder">Could not run this model: ' +
        esc(r.error) +
        "</p>";
      return;
    }
    var m = r.model;
    var board = m.board_throughput_img_s;
    var mine = r.latencyMs;
    var perImg = 1000 / mine;

    $("output").innerHTML =
      '<div class="out-head"><h3>' +
      esc(m.predictedLabel || r.predictedLabel) +
      "</h3>" +
      verdict(r.predicted, state.truth) +
      "</div>" +
      '<p class="out-note">' +
      "confidence " +
      pct(r.confidence, 1) +
      (state.truth !== null
        ? " · true class is <strong>" +
          esc(state.results.classes[state.truth]) +
          "</strong>"
        : "") +
      "</p>" +
      probabilityBars(r.ranked, state.truth) +
      '<div class="latency">' +
      '<div><div class="k">Median latency</div><div class="v">' +
      mine.toFixed(3) +
      " ms</div></div>" +
      '<div><div class="k">Fastest of ' +
      r.timedRuns +
      '</div><div class="v">' +
      r.fastestMs.toFixed(3) +
      " ms</div></div>" +
      '<div><div class="k">Single-image rate</div><div class="v">' +
      (perImg >= 10 ? perImg.toFixed(0) : perImg.toFixed(1)) +
      " img/s</div></div>" +
      '<div><div class="k">Batch on PYNQ-Z1</div><div class="v">' +
      (board ? fmtThroughput(board) + " img/s" : "—") +
      "</div></div>" +
      '<div><div class="k">Model load</div><div class="v">' +
      Math.round(r.loadMs) +
      " ms</div></div>" +
      "</div>" +
      '<p class="out-note" style="margin-top:16px">' +
      "Measured in this browser with onnxruntime-web " +
      FF.inference.version +
      " (WASM, one thread): " +
      r.timedRuns +
      " timed runs after 3 warm-ups, median reported" +
      (r.batch > 1
        ? ", each run repeating the inference " +
          r.batch +
          "× back to back and dividing, " +
          "because the browser clock is coarser than a sub-millisecond model"
        : "") +
      ". The PYNQ column is batch-256 throughput on a dual-core Cortex-A9. " +
      "Browser timing and board batch throughput use different measurement conditions.</p>";
  }

  function renderComparison(results) {
    var truth = state.truth;

    /* Median-of-12 latency, but the compact rows land at 0.3-0.5 ms where the
     * browser timer's own resolution dominates: they tie, and they swap order
     * between runs. So the table makes no claim about a winner. What does repeat
     * is how far the slowest row sits from the fastest, so that is the headline. */
    var minMs = null,
      maxMs = null,
      slowest = null;
    results.forEach(function (r) {
      if (r.error) return;
      if (minMs === null || r.latencyMs < minMs) minMs = r.latencyMs;
      if (maxMs === null || r.latencyMs > maxMs) {
        maxMs = r.latencyMs;
        slowest = r;
      }
    });
    var spread =
      minMs !== null && maxMs !== null && minMs > 0 ? maxMs / minMs : null;

    var rows = results
      .map(function (r) {
        var m = r.model;
        if (r.error) {
          return (
            '<tr><td class="mono">' +
            esc(m.id) +
            '</td><td colspan="8" class="pend">' +
            esc(r.error) +
            "</td></tr>"
          );
        }
        var v =
          truth === null
            ? '<span class="pend">—</span>'
            : r.predicted === truth
              ? '<span class="ok">correct</span>'
              : '<span class="no">wrong</span>';
        return (
          "<tr" +
          (m.id === state.modelId ? ' class="sel"' : "") +
          ">" +
          '<td class="mono">' +
          esc(m.id) +
          "</td>" +
          "<td>" +
          pct(m.paper_accuracy) +
          "</td>" +
          '<td class="dim2">' +
          fmtBytes(m.bytes) +
          "</td>" +
          '<td class="dim2">' +
          fmtInt(m.params) +
          "</td>" +
          '<td class="mono">' +
          esc(r.predictedLabel) +
          "</td>" +
          "<td>" +
          pct(r.confidence, 1) +
          "</td>" +
          "<td>" +
          v +
          "</td>" +
          "<td><strong>" +
          r.latencyMs.toFixed(3) +
          "</strong> ms</td>" +
          "<td>" +
          fmtThroughput(m.board_throughput_img_s) +
          "</td>" +
          "</tr>"
        );
      })
      .join("");

    $("compareBlock").hidden = false;
    $("compareBlock").innerHTML =
      '<div class="out-head"><h3>All five checkpoints, one image</h3>' +
      (spread && spread >= 1.5 && slowest
        ? '<span class="out-verdict ok">' +
          esc(slowest.model.id) +
          " took " +
          spread.toFixed(0) +
          "× the fastest row</span>"
        : "") +
      "</div>" +
      '<p class="out-note">Same image and browser. Each model uses its own training-time resize and normalization.</p>' +
      '<div class="cmp-scroll"><table class="cmp"><thead><tr>' +
      "<th>Model</th><th>Paper acc</th><th>Size</th><th>Params</th>" +
      "<th>Predicted</th><th>Score</th><th>Result</th><th>Browser ms</th><th>Board img/s</th>" +
      "</tr></thead><tbody>" +
      rows +
      "</tbody></table></div>" +
      '<p class="out-note" style="margin-top:16px">' +
      "Browser timings vary with hardware and load. Small differences below one millisecond may be timer noise. Board results are historical measurements at batch size 256.</p>";
    $("compareBlock").scrollIntoView({ block: "nearest" });
  }

  /* ------------------------------------------------------------ actions */

  function setBusy(on, msg) {
    state.busy = on;
    $("runOne").disabled = on;
    $("runAll").disabled = on;
    $("upload").disabled = on;
    $("file").disabled = on;
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
    setBusy(
      true,
      m.lazy ? "Downloading " + fmtBytes(m.bytes) + " checkpoint…" : "Running…",
    );
    $("output").innerHTML =
      '<p class="placeholder">Running ' + esc(m.id) + "…</p>";
    FF.inference
      .run(state.image, m, state.results.classes)
      .then(function (r) {
        renderSingle(r);
        setBusy(false, "Done.");
      })
      .catch(function (err) {
        $("output").innerHTML = '<p class="placeholder">Inference failed.</p>';
        setBusy(false, "");
        fail(err);
      });
  }

  function predictAll() {
    if (state.busy || !state.image) return;
    setBusy(
      true,
      "Running all five… this downloads about 18 MB the first time.",
    );
    $("output").innerHTML =
      '<p class="placeholder">Running every checkpoint sequentially…</p>';
    var models = state.results.demo_models;
    FF.inference
      .runAll(state.image, models, state.results.classes, function (i, m) {
        $("status").textContent =
          "Running " +
          (i + 1) +
          "/" +
          models.length +
          " · " +
          m.id +
          (m.lazy ? " (" + fmtBytes(m.bytes) + " download)" : "");
      })
      .then(function (results) {
        var selected = results.find(function (r) {
          return r.model.id === state.modelId;
        });
        if (selected) renderSingle(selected);
        renderComparison(results);
        setBusy(false, "Done. " + results.length + " checkpoints executed.");
      })
      .catch(function (err) {
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

  function chartName(id) {
    return CHART_NAME[id] || id;
  }

  function renderFrontier() {
    var pts = state.results.frontier.filter(function (p) {
      return p.family === "cpu";
    });
    var x = function (v) {
      return 90 + ((v - 100) / 400) * 760;
    };
    var y = function (v) {
      return 315 - ((v - 90) / 3) * 240;
    };
    var parts = [];
    [90, 91, 92, 93].forEach(function (v) {
      parts.push(
        '<line x1="90" y1="' +
          y(v) +
          '" x2="850" y2="' +
          y(v) +
          '" stroke="var(--line)"/>',
      );
      parts.push(
        '<text x="70" y="' +
          (y(v) + 4) +
          '" text-anchor="end" fill="var(--muted)" font-size="13">' +
          v +
          "%</text>",
      );
    });
    [100, 200, 300, 400, 500].forEach(function (v) {
      parts.push(
        '<text x="' +
          x(v) +
          '" y="346" text-anchor="middle" fill="var(--muted)" font-size="13">' +
          v +
          "</text>",
      );
    });
    parts.push(
      '<text x="470" y="385" text-anchor="middle" fill="var(--muted)" font-size="13">PYNQ-Z1 throughput · images / second</text>',
    );
    parts.push(
      '<text x="90" y="35" fill="var(--muted)" font-size="13">Test accuracy · paper records</text>',
    );
    parts.push(
      '<polyline points="' +
        pts
          .map(function (p) {
            return x(p.throughput) + "," + y(p.accuracy);
          })
          .join(" ") +
        '" fill="none" stroke="var(--cpu)" stroke-width="2"/>',
    );
    pts.forEach(function (p, i) {
      var dx = i === 2 ? -12 : 14;
      var anchor = i === 2 ? "end" : "start";
      parts.push(
        '<circle cx="' +
          x(p.throughput) +
          '" cy="' +
          y(p.accuracy) +
          '" r="6" fill="var(--cpu)" stroke="var(--bg)" stroke-width="3"/>',
      );
      parts.push(
        '<text x="' +
          (x(p.throughput) + dx) +
          '" y="' +
          (y(p.accuracy) - 18) +
          '" text-anchor="' +
          anchor +
          '" fill="var(--ink)" font-size="13" font-weight="600">' +
          esc(chartName(p.label)) +
          "</text>",
      );
      parts.push(
        '<text x="' +
          (x(p.throughput) + dx) +
          '" y="' +
          (y(p.accuracy) + 1) +
          '" text-anchor="' +
          anchor +
          '" fill="var(--muted)" font-size="11">' +
          p.accuracy.toFixed(2) +
          "% · " +
          p.throughput +
          " img/s</text>",
      );
    });
    $("frontierSvg").innerHTML = parts.join("");
  }

  var ROUTE_LABEL = {
    teacher: "Teacher / baseline",
    distill: "Distillation",
    quant: "Quantisation",
    cpu: "CPU INT8",
    bnn: "BNN",
    fpga: "FPGA",
  };

  /* Short forms for the table's route chip: eleven columns leave no room for
   * the filter-button wording. */
  var ROUTE_CHIP = {
    teacher: "Teacher",
    distill: "Distill",
    quant: "Quant",
    cpu: "CPU INT8",
    bnn: "BNN",
    fpga: "FPGA",
  };

  function renderResultsTable() {
    var rows = state.results.table;
    if (state.filter !== "all") {
      rows = rows.filter(function (r) {
        return r.route === state.filter;
      });
    }
    var body = rows
      .map(function (r) {
        var fam =
          r.route === "cpu"
            ? "cpu"
            : r.route === "bnn" || r.route === "fpga"
              ? "fabric"
              : "";
        return (
          "<tr>" +
          '<td><span class="chip ' +
          fam +
          '">' +
          esc(ROUTE_CHIP[r.route] || r.route) +
          "</span></td>" +
          '<td class="mono">' +
          esc(r.model) +
          "</td>" +
          "<td>" +
          esc(r.precision) +
          "</td>" +
          "<td>" +
          (r.params ? fmtInt(r.params) : '<span class="dim">—</span>') +
          "</td>" +
          "<td>" +
          (r.val_accuracy
            ? pct(r.val_accuracy)
            : '<span class="dim">—</span>') +
          "</td>" +
          "<td>" +
          (r.test_accuracy
            ? pct(r.test_accuracy)
            : '<span class="dim">—</span>') +
          "</td>" +
          "<td>" +
          (r.board_throughput_img_s
            ? fmtThroughput(r.board_throughput_img_s)
            : '<span class="dim">—</span>') +
          "</td>" +
          "<td>" +
          (r.board_total_time_s
            ? r.board_total_time_s.toFixed(1)
            : '<span class="dim">—</span>') +
          "</td>" +
          "<td>" +
          esc(r.device) +
          "</td>" +
          "<td>" +
          (r.deliverable === "yes"
            ? '<span class="chip yes">yes</span>'
            : '<span class="dim">no</span>') +
          "</td>" +
          '<td class="mono">' +
          esc(r.source) +
          "</td>" +
          "</tr>"
        );
      })
      .join("");

    $("resultsBody").innerHTML =
      body ||
      '<tr><td colspan="11" class="dim">Nothing in this route.</td></tr>';
    $("resultsCount").textContent =
      rows.length + " of " + state.results.table.length + " rows";
  }

  /* ------------------------------------------------------------ wiring */

  function wire() {
    $("runOne").addEventListener("click", predictOne);
    $("runAll").addEventListener("click", predictAll);

    var upload = $("upload");
    var file = $("file");
    upload.addEventListener("click", function () {
      file.click();
    });
    file.addEventListener("change", function () {
      var f = file.files && file.files[0];
      if (!f || state.busy) return;
      if (f.size > 8 * 1024 * 1024) {
        fail(new Error("Choose an image smaller than 8 MiB."));
        return;
      }
      var url = URL.createObjectURL(f);
      var img = new Image();
      img.onload = function () {
        URL.revokeObjectURL(url);
        state.image = img;
        state.truth = null;
        state.sampleIndex = -1;
        renderSamples();
        resetOutput();
        $("status").textContent =
          "Loaded " +
          f.name +
          ". No ground truth for an upload, so correctness is not shown.";
      };
      img.onerror = function () {
        URL.revokeObjectURL(url);
        fail(new Error("This image could not be decoded."));
      };
      img.src = url;
    });

    var filters = ["all", "teacher", "distill", "quant", "cpu", "bnn", "fpga"];
    $("filters").innerHTML = filters
      .map(function (f) {
        return (
          '<button type="button" data-f="' +
          f +
          '" class="' +
          (f === "all" ? "on" : "") +
          '">' +
          (f === "all" ? "All routes" : ROUTE_LABEL[f] || f) +
          "</button>"
        );
      })
      .join("");
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
      fetch("data/results.json").then(function (r) {
        if (!r.ok) throw new Error("Results request failed: " + r.status);
        return r.json();
      }),
    ])
      .then(function (both) {
        state.results = both[0];

        /* Default to the balanced tier, not the teacher: it is the model this
         * project actually recommends, and it is 184 KB instead of 15 MB. */
        var preferred =
          modelById("tinyfast_xxs") || state.results.demo_models[0];
        state.modelId = preferred.id;

        renderModels();
        renderSamples();
        renderFrontier();
        renderResultsTable();
        wire();
        selectSample(0);

        /* Warm the default checkpoint so the first click is instant. Failures are
         * silent here - the button will surface them with a real message. */
        FF.inference.load(preferred).catch(function () {});
        if (new URLSearchParams(location.search).has("selftest")) FF.selftest();
      })
      .catch(function (err) {
        document.getElementById("output").innerHTML =
          '<p class="placeholder">Could not load the data files: ' +
          esc(err.message || err) +
          "</p>";
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
    function log(s) {
      lines.push(s);
      out.textContent = lines.join("\n");
    }

    function loadImg(src, attempt) {
      attempt = attempt || 0;
      return new Promise(function (res, rej) {
        var i = new Image();
        i.onload = function () {
          res(i);
        };
        i.onerror = function () {
          if (attempt < 2) {
            setTimeout(
              function () {
                res(loadImg(src, attempt + 1));
              },
              150 * (attempt + 1),
            );
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
    var hits = 0,
      tries = 0;
    var started = Date.now();

    log(
      "ort " +
        FF.inference.version +
        " models " +
        models.length +
        " samples " +
        samples.length,
    );

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

    warm
      .then(function () {
        log("images loaded " + Object.keys(images).length);
        var chain = Promise.resolve();
        models.forEach(function (m) {
          chain = chain.then(function () {
            var t = Date.now();
            return FF.inference
              .load(m)
              .then(function () {
                log("load ok " + m.id + " " + (Date.now() - t) + "ms");
              })
              .catch(function (e) {
                log("load FAIL " + m.id + " " + (e.message || e));
              });
          });
          samples.forEach(function (s, i) {
            chain = chain.then(function () {
              return FF.inference
                .run(images[s.file], m, state.results.classes, 0)
                .then(function (r) {
                  tries += 1;
                  var ok = r.predicted === s.label_index;
                  if (ok) hits += 1;
                  log(
                    "run " +
                      m.id +
                      " s" +
                      i +
                      " " +
                      (ok ? "OK  " : "MISS") +
                      " truth=" +
                      s.label +
                      " pred=" +
                      r.predictedLabel +
                      " conf=" +
                      r.confidence.toFixed(3) +
                      " ms=" +
                      r.latencyMs.toFixed(3),
                  );
                })
                .catch(function (e) {
                  tries += 1;
                  log("run " + m.id + " s" + i + " ERR " + (e.message || e));
                });
            });
          });
        });
        return chain;
      })
      .then(function () {
        var secs = ((Date.now() - started) / 1000).toFixed(1);
        log("TOTAL " + hits + "/" + tries + " correct in " + secs + "s");
        log("SELFTEST DONE");
        document.title = "SELFTEST " + hits + "/" + tries;
      })
      .catch(function (e) {
        log("FATAL " + (e.message || e));
        log("SELFTEST DONE");
        document.title = "SELFTEST ERROR";
      });
  };

  FF.site = { state: state, boot: boot };
  document.addEventListener("DOMContentLoaded", boot);
})(window.FF);
