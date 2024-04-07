let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);

  // In the space below create a neural network that can classify hand gestures
  // corresponding to rock, paper, scissors, lizard, and spock. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.
  // HINT: Take a look at the Rock-Paper-Scissors example. We also suggest
  // using ReLu activation functions where applicable.
  model = tf.sequential({
    layers: [
      // Flatten the output from the pre-trained MobileNet model
      tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
      // Add a dense hidden layer with 128 units and ReLU activation
      tf.layers.dense({ units: 128, activation: 'relu' }),
      // Add another dense hidden layer with 64 units and ReLU activation
      tf.layers.dense({ units: 64, activation: 'relu' }),
      // Output layer with 5 units for the 5 classes and softmax activation
      tf.layers.dense({ units: 5, activation: 'softmax' })
    ]
  });

  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001);

  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  let loss = 0;
  // Increase the number of epochs to 20 for better training
  model.fit(dataset.xs, dataset.ys, {
    epochs: 20,
    validationSplit: 0.2,  // Use 20% of the data for validation
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log(`Epoch ${epoch + 1}, Loss: ${loss}, Accuracy: ${logs.acc.toFixed(5)}, Validation Accuracy: ${logs.val_acc.toFixed(5)}`);
      }
    }
  });
}


function handleButton(elem) {
  switch (elem.id) {
    case "0":
      rockSamples++;
      document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
      break;
    case "3":
      spockSamples++;
      document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
      break;
    case "4":
      lizardSamples++;  // Add a case for lizard samples.
      document.getElementById("lizardsamples").innerText = "Lizard samples:" + lizardSamples;
      break;
  }
  label = parseInt(elem.id);
  const img = webcam.capture();
  dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "I see Rock";
        break;
      case 1:
        predictionText = "I see Paper";
        break;
      case 2:
        predictionText = "I see Scissors";
        break;
      case 3:
        predictionText = "I see Spock";
        break;
      case 4:
        predictionText = "I see Lizard";  // Add a case for lizard samples.
        break;
    }
    document.getElementById("prediction").innerText = predictionText;
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

function doTraining() {
  train();
  alert("Training Done!");
}

function startPredicting() {
  isPredicting = true;
  predict();
}

function stopPredicting() {
  isPredicting = false;
  predict();
}

function saveModel() {
  model.save('downloads://my_model');
}

async function init() {
  await webcam.setup();
  mobilenet = await loadMobilenet();
  tf.tidy(() => mobilenet.predict(webcam.capture()));
}

init();
