const createCsvWriter = require('csv-writer').createObjectCsvWriter;

writePredictions = (filePath, headers, records) => {
    csvWriter = createCsvWriter({
        path: filePath,
        header: headers
    });

    return csvWriter.writeRecords(records);
}

module.exports = writePredictions;