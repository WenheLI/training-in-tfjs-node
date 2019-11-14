const tf = require('@tensorflow/tfjs-node');
const writePredictions = require('../utils');

async function run() {
    const model = await tf.loadLayersModel(`file://./models/mnist/model.json`);
    const results = [];
    const header = [
        {id: 'id', title: 'ImageId'},
        {id: 'label', title: 'Label'}
    ];
    const testData = tf.data.csv(
        'file://./data/mnist/test.csv', {
            hasHeader: true
        });
    const processedTest = testData.map((x) => tf.tensor(Object.values(x), [28, 28, 1]));
    let index = 1;
    await processedTest.forEachAsync((d) => {
        const res = model.predict(d.reshape([1, 28, 28, 1]));
        const {indices} = tf.topk(res);
        results.push({id: index, label: indices.dataSync()[0]});
        index += 1;
    });

    writePredictions('./results/mnist.csv', header, results)
    
}

run()