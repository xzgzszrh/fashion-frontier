/* Real inference, in the browser, on the paper's actual checkpoints.
 *
 * Nothing here is a screenshot or a cached result. The model files in
 * web/models/ were re-verified against the full 10 000-image Fashion-MNIST test
 * set before being shipped, and running them here reproduces the predictions
 * that the PYNQ-Z1 produced - on different hardware, through a different
 * runtime, which is exactly the point of the comparison.
 *
 * onnxruntime-web is loaded from a CDN by index.html as a classic script, so
 * `window.ort` already exists by the time this file runs.
 */
window.FF = window.FF || {};

(function (FF) {
  "use strict";

  // Pinned. The models were verified against this build, and a silent minor
  // bump is not something a benchmark page should accept.
  var ORT_VERSION = "1.22.0";
  var ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";

  var sessions = new Map();
  var configured = false;

  function ort() {
    if (!window.ort) {
      throw new Error(
        "onnxruntime-web did not load. It is fetched from a CDN, so this needs a network connection."
      );
    }
    if (!configured) {
      window.ort.env.wasm.wasmPaths = ORT_BASE;
      // GitHub Pages cannot send COOP/COEP headers, so SharedArrayBuffer is
      // unavailable and threaded wasm would fail to start. One thread it is.
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.logLevel = "error";
      configured = true;
    }
    return window.ort;
  }

  function session(model) {
    var key = model.file;
    if (!sessions.has(key)) {
      var pending = ort()
        .InferenceSession.create(model.file, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        })
        .catch(function (err) {
          sessions.delete(key);
          throw err;
        });
      sessions.set(key, pending);
    }
    return sessions.get(key);
  }

  /* Mirror of the training-time preprocessing. The normalisation constants come
   * from results.json, which derives them from the configs that produced the
   * paper's numbers - not from a guess. */
  function toTensor(image, model) {
    var size = model.image_size;
    var channels = model.channels;

    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("this browser refused a 2D canvas context");

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(image, 0, 0, size, size);

    var px = ctx.getImageData(0, 0, size, size).data;
    var plane = size * size;
    var out = new Float32Array(channels * plane);

    for (var y = 0; y < size; y += 1) {
      for (var x = 0; x < size; x += 1) {
        var i = (y * size + x) * 4;
        // Fashion-MNIST is greyscale; the RGB branch replicates one luma plane
        // into three channels rather than inventing colour.
        var value = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255;
        for (var c = 0; c < channels; c += 1) {
          var mean = model.mean[c] === undefined ? model.mean[0] : model.mean[c];
          var std = model.std[c] === undefined ? model.std[0] : model.std[c];
          out[c * plane + y * size + x] = (value - mean) / std;
        }
      }
    }
    return new (ort().Tensor)("float32", out, [1, channels, size, size]);
  }

  function softmax(logits) {
    var max = -Infinity;
    var i;
    for (i = 0; i < logits.length; i += 1) if (logits[i] > max) max = logits[i];
    var sum = 0;
    var exps = new Array(logits.length);
    for (i = 0; i < logits.length; i += 1) {
      exps[i] = Math.exp(logits[i] - max);
      sum += exps[i];
    }
    for (i = 0; i < exps.length; i += 1) exps[i] /= sum;
    return exps;
  }

  function median(values) {
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /* Warm-up matters: the first run pays for kernel selection and memory
   * allocation, and it is routinely 3-10x slower than steady state. Reporting
   * the first run would make the numbers meaningless. */
  var WARMUP = 3;
  var TIMED = 12;
  /* performance.now() is quantised - Chrome clamps it to ~100 us - so timing one
   * 0.3 ms inference measures the clock rather than the model, and the median
   * snaps between 0.3 and 0.5 ms from run to run. Each timed sample therefore
   * repeats the inference BATCH times back to back and divides. The warm-up runs
   * are reused as the probe that sizes BATCH, so nothing is measured twice. */
  var MIN_SAMPLE_MS = 5;
  var MAX_BATCH = 256;

  FF.inference = {
    version: ORT_VERSION,

    load: function (model) {
      return session(model).then(function () { return true; });
    },

    /* Run one image through one model and report both the prediction and a
     * steady-state latency distribution. */
    run: function (image, model, classes, samples) {
      samples = samples || TIMED;
      var t0 = performance.now();
      return session(model).then(function (sess) {
        var tensor = toTensor(image, model);
        var feeds = {};
        feeds[sess.inputNames[0]] = tensor;
        var outName = sess.outputNames[0];
        var loadMs = performance.now() - t0;
        var times = [];
        var logits = null;
        var batch = 1;
        var probe = 0;

        function once() {
          return sess.run(feeds).then(function (out) {
            if (logits === null) logits = Array.prototype.slice.call(out[outName].data);
            return out;
          });
        }

        function warm(n) {
          if (n >= WARMUP) return Promise.resolve();
          var t = performance.now();
          return once().then(function () {
            probe = performance.now() - t;
            return warm(n + 1);
          });
        }

        /* One timed sample: `batch` inferences, divided down to a per-image
         * figure. */
        function sample() {
          var t = performance.now();
          var chain = Promise.resolve();
          for (var k = 0; k < batch; k += 1) chain = chain.then(once);
          return chain.then(function () { times.push((performance.now() - t) / batch); });
        }

        function timed(n) {
          if (n >= samples) return Promise.resolve();
          return sample().then(function () { return timed(n + 1); });
        }

        return warm(0).then(function () {
          if (probe > 0) {
            batch = Math.max(1, Math.min(MAX_BATCH, Math.ceil(MIN_SAMPLE_MS / probe)));
          }
          return timed(0);
        }).then(function () {
          var probs = softmax(logits);
          var ranked = probs
            .map(function (p, idx) { return { index: idx, label: classes[idx], probability: p }; })
            .sort(function (a, b) { return b.probability - a.probability; });
          return {
            model: model,
            ranked: ranked,
            predicted: ranked[0].index,
            predictedLabel: ranked[0].label,
            confidence: ranked[0].probability,
            latencyMs: median(times),
            fastestMs: Math.min.apply(null, times),
            slowestMs: Math.max.apply(null, times),
            loadMs: loadMs,
            timedRuns: times.length,
            batch: batch,
          };
        });
      });
    },

    /* Sequential on purpose: running five sessions concurrently on one wasm
     * thread just interleaves them and muddies every latency figure. */
    runAll: function (image, models, classes, onProgress) {
      var results = [];
      var chain = Promise.resolve();
      models.forEach(function (model, i) {
        chain = chain.then(function () {
          if (onProgress) onProgress(i, model);
          return FF.inference.run(image, model, classes).then(function (r) {
            results.push(r);
            return r;
          }).catch(function (err) {
            results.push({ model: model, error: err.message || String(err) });
            return null;
          });
        });
      });
      return chain.then(function () { return results; });
    },
  };
})(window.FF);
