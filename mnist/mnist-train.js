const tf = require('@tensorflow/tfjs-node');
const model = require('./mnist-model');
async function run() {
    model.summary()
    const csvDataset = tf.data.csv(
        'file://./data/mnist/train.csv', {
            hasHeader: true, 
            columnConfigs: {
                label: {
                    isLabel: true
                }
            }
        });
    const flattenedDataset = csvDataset.map(({
        xs,
        ys
    }) => {
        return {
            xs: tf.tensor(Object.values(xs), [28, 28, 1]),
            ys: tf.oneHot((Object.values(ys)[0]), 10)
        };
    }).shuffle(1000).batch(64);
    await model.fitDataset((flattenedDataset), {
        epochs: 30,
        callbacks: tf.node.tensorBoard('./logdir', {
            updateFreq: 'batch'
        })
    });

    model.save('file://./fist')

    

}
run();