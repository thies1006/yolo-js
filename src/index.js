import * as tf from '@tensorflow/tfjs';

async function getPesos() {

  const pool4 = (await fetch('/pool_4.txt').then(res => res.text())).split(',');
  const w_conv_5_parte1 = (await fetch('/w_conv_5_parte1.txt').then(res => res.text())).split(',');
  const w_conv_5_parte2 = (await fetch('/w_conv_5_parte2.txt').then(res => res.text())).split(',');

  const pool_4_tensor = tf.tensor1d(pool4)
  const w_conv_5_tensor = tf.tensor1d(w_conv_5_parte1.concat(w_conv_5_parte2))

  document.getElementById("input").innerHTML = tf.reshape(pool_4_tensor, [1,30,30,88])
  document.getElementById("weights").innerHTML = tf.reshape(w_conv_5_tensor, [3,3,88,176])

  document.getElementById("demo").innerHTML = tf.conv2d(tf.reshape(pool_4_tensor, [1,30,30,88]),tf.reshape(w_conv_5_tensor, [3,3,88,176]),[1,1, 1, 1], 'same');

  tf.print(tf.conv2d(tf.reshape(pool_4_tensor, [1,30,30,88]),tf.reshape(w_conv_5_tensor, [3,3,88,176]),[1,1, 1, 1], 'same')) //Esta de aquí es para parar el código

  }

getPesos()