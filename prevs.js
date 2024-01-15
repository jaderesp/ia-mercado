const tf = require('@tensorflow/tfjs');
const axios = require('axios');
const yahoo = require('yahoo-finance-webscraper');

const empresa = 'FB';
const inicio = '2012-01-01';
const final = '2020-01-01';
const previsaoDias = 60;

// Função para obter dados do Yahoo Finance
async function obterDadosYahooFinance(empresa, inicio, final) {
    const url = `https://query1.finance.yahoo.com/v7/finance/download/${empresa}?period1=${new Date(inicio).getTime() / 1000}&period2=${new Date(final).getTime() / 1000}&interval=1d&events=history`;

    try {
        const resposta = await axios.get(url);
        return resposta.data;
    } catch (erro) {
        console.error('Erro ao obter dados do Yahoo Finance:', erro.message);
        throw erro;
    }
}

// Função para normalizar dados
function normalizarDados(dados) {
    const min = tf.min(dados);
    const max = tf.max(dados);

    return {
        normalizados: tf.div(tf.sub(dados, min), tf.sub(max, min)),
        min,
        max
    };
}

// Função para preparar dados de treinamento
function prepararDadosTreinamento(dadosNormalizados, previsaoDias) {
    const xTreinar = [];
    const yTreinar = [];

    for (let i = previsaoDias; i < dadosNormalizados.length; i++) {
        xTreinar.push(dadosNormalizados.slice(i - previsaoDias, i));
        yTreinar.push(dadosNormalizados[i]);
    }

    return {
        xTreinar: tf.tensor(xTreinar),
        yTreinar: tf.tensor(yTreinar)
    };
}

// Função para construir o modelo LSTM
function construirModelo(previsaoDias) {
    const modelo = tf.sequential();

    modelo.add(tf.layers.lstm({ units: 50, returnSequences: true, inputShape: [previsaoDias, 1] }));
    modelo.add(tf.layers.dropout({ rate: 0.2 }));
    modelo.add(tf.layers.lstm({ units: 50, returnSequences: true }));
    modelo.add(tf.layers.dropout({ rate: 0.2 }));
    modelo.add(tf.layers.lstm({ units: 50 }));
    modelo.add(tf.layers.dropout({ rate: 0.2 }));
    modelo.add(tf.layers.dense({ units: 1 }));

    modelo.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    return modelo;
}

// Função para treinar o modelo
async function treinarModelo(modelo, xTreinar, yTreinar, epochs, batchSize) {
    return await modelo.fit(xTreinar, yTreinar, {
        epochs,
        batchSize
    });
}

// Função para fazer previsões
function fazerPrevisoes(modelo, dadosEntrada, normalizacao) {
    const xTeste = [];

    for (let i = previsaoDias; i < dadosEntrada.length; i++) {
        xTeste.push(dadosEntrada.slice(i - previsaoDias, i));
    }

    const previsaoTensor = modelo.predict(tf.tensor(xTeste));
    const previsaoDesnormalizada = tf.mul(tf.add(previsaoTensor, normalizacao.min), tf.sub(normalizacao.max, normalizacao.min));

    return previsaoDesnormalizada.arraySync();
}

// Função principal
async function main() {
    const dadosCsv = await obterDadosYahooFinance(empresa, inicio, final);

    const dados = tf.data.csv.parseCsv(dadosCsv, { columnConfigs: { Date: { isDate: true } } });

    const dadosFechamento = dados.map(row => row.Close).slice(previsaoDias);

    const { normalizados, min, max } = normalizarDados(dadosFechamento);

    const { xTreinar, yTreinar } = prepararDadosTreinamento(normalizados.arraySync(), previsaoDias);

    const modelo = construirModelo(previsaoDias);

    await treinarModelo(modelo, xTreinar, yTreinar, 25, 32);

    const dadosTeste = await obterDadosYahooFinance(empresa, final, new Date().toISOString().split('T')[0]);

    const precosReais = tf.tensor(dadosTeste.map(row => row.Close));
    const totalDados = tf.concat([dadosFechamento, precosReais], 0);

    const modeloEntrada = totalDados.slice(totalDados.shape[0] - dadosTeste.length - previsaoDias).reshape([1, -1, 1]);

    const modeloEntradaNormalizado = tf.div(tf.sub(modeloEntrada, min), tf.sub(max, min));

    const previsaoPrecos = fazerPrevisoes(modelo, modeloEntradaNormalizado.arraySync(), { min, max });

    console.log('Previsão dos preços:', previsaoPrecos);
}

main();
