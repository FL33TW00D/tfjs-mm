import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgpu';
tf.setBackend('webgpu').then(() => main());

function main() {
    const M = parseInt(document.getElementById("selM").value, 10);
    const N = parseInt(document.getElementById("selN").value, 10);
    const K = parseInt(document.getElementById("selK").value, 10);

    document.getElementById('benchmarkButton').addEventListener('click', () => benchmark(M, N, K));
}

function benchmark(M, N, K) {
    // Warm up
    let a = tf.randomUniform([M, K], 0.5, 0.6);
    let b = tf.randomUniform([K, N], 0.5, 0.6);
    const c = tf.matMul(a, b);

    let totalTime = 0;
    for (let i = 0; i < 10; i++) {
        let a = tf.randomUniform([M, K], 0.5, 0.6);
        let b = tf.randomUniform([K, N], 0.5, 0.6);
        let start = performance.now();
        const c = tf.matMul(a, b);
        let end = performance.now();
        totalTime += end - start;
    }
    console.log("Total time: " + totalTime);
    console.log("Average time: " + totalTime / 10);
    let flops = M * N * K * 2 * 10;
    let gflops = flops / totalTime / 1e6;
    document.getElementById("result").innerHTML = "GFLOPS: " + gflops; 
}
