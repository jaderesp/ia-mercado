const express = require('express');
const tf = require('@tensorflow/tfjs');
const fs = require('fs').promises;

const app = express();
const port = 9000;

app.use(express.json());

async function carregarDados() {
    try {
        const dadosRaw = await fs.readFile('treinamento.json', 'utf8');
        const dados = JSON.parse(dadosRaw);
        return dados;
    } catch (erro) {
        console.error('Erro ao carregar dados:', erro);
        throw erro;
    }
}

async function salvarDados(dados) {
    try {
        const dadosJSON = JSON.stringify(dados, null, 2);
        await fs.writeFile('treinamento.json', dadosJSON, 'utf8');
    } catch (erro) {
        console.error('Erro ao salvar dados:', erro);
        throw erro;
    }
}

async function treinarModelo(dadosTreinamento) {
    const modelo = tf.sequential();
    modelo.add(tf.layers.dense({ units: 64, inputShape: [dadosTreinamento.entrada[0].length], activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    modelo.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    modelo.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    console.log(dadosTreinamento.entrada)

    try{

    const entradas = tf.tensor2d(dadosTreinamento.entrada).print();
    const rotulos = tf.tensor2d(dadosTreinamento.rotulo);

    await modelo.fit(entradas, rotulos, { epochs: 50 });

    }catch(error){
        console.log("\r\n Ocorreu um erro ao executar o tensorFlow.js: ", error)
    }

    return modelo;
}

function fazerPrevisoes(modelo, novosDados) {
    const entradas = tf.tensor2d(novosDados);
    const previsoes = modelo.predict(entradas);
    return previsoes.dataSync();
}

app.post('/treinarPrever', async (req, res) => {
    try {
        const dados = await carregarDados();

        const modeloTreinado = await treinarModelo(dados.dadosTreinamento);

        // Converta os novos dados para um array multidimensional
        const novosDados = tf.tensor2d(req.body.novosDados.map(dado => [...dado]));

        const resultadoPrevisoes = fazerPrevisoes(modeloTreinado, novosDados.arraySync());

        // Adicione os novos resultados ao histórico
        dados.dadosTreinamento.entrada.push(...novosDados.arraySync());
        dados.dadosTreinamento.rotulo.push(...resultadoPrevisoes.map(valor => Math.round(valor)));

        // Atualize os novos dados no arquivo JSON
        await salvarDados(dados);

        res.json({ resultadoPrevisoes });
    } catch (erro) {
        console.error('Erro na rota /treinarPrever:', erro);
        res.status(500).json({ erro: 'Erro interno no servidor' });
    }
});

app.listen(port, () => {
    console.log(`API rodando em http://localhost:${port}`);
});
