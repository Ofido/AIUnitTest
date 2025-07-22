# Initial Project Documentation

Esta seção descreve a ideia inicial, objetivos e visão geral arquitetural do projeto AIUnitTest.

1. Objetivo

   O Auto Test Updater busca automatizar a geração e atualização de testes unitários em projetos Python,
   reduzindo o trabalho manual e melhorando a cobertura de forma contínua.

2. Contexto

   Muitas equipes enfrentam desafios para manter a cobertura de testes alta. Testes faltantes ou desatualizados
   podem levar a regressões e dificultar refatorações. Ao integrar um LLM, podemos:

   - Detectar rapidamente linhas não testadas.
   - Gerar casos de teste coerentes com o estilo do projeto.
   - Atualizar testes existentes quando a lógica do código muda.

3. Principais Componentes

   - CLI (cli.py)
     - Parse de argumentos e carregamento de configurações do pyproject.toml.
     - Modo manual vs. --auto.
   - Coverage Helper (coverage_helper.py)
     - Uso da API do Coverage.py para coletar linhas sem cobertura.
     - Retorna mapeamento arquivo → linhas faltantes.
   - File Helper (file_helper.py)
     - Localização de módulos fonte e arquivos de teste.
     - Leitura e escrita de arquivos.
   - LLM Integration (llm.py)
     - Conexão assíncrona com OpenAI GPT para gerar ou completar testes.
     - Montagem de prompts (system + user).
   - Orquestração (main.py)
     - Fluxo principal em asyncio: coleta cobertura, encontra testes, chama LLM e grava resultados.
     - Logs e tratamento de erros.

4. Fluxo de Execução

   ```mermaid
   flowchart TD
       A[Iniciar CLI] --> B{--auto ?}
       B -- sim --> C[Carregar config do pyproject.toml]
       B -- não --> D[Usar args passados]
       D --> E[Validar paths]
       C --> E
       E --> F[Collect Missing Lines]
       F --> G{Arquivos faltando?}
       G -- não --> H[Fim]
       G -- sim --> I[Para cada arquivo]
       I --> J[Find Test File]
       J --> K[Read source & test]
       K --> L[Chama LLM Async]
       L --> M[Escreve arquivo de teste]
       M --> I
       M --> H
   ```

5. Critérios de Sucesso

   - Cobertura mínima de X% após execução.
   - Testes gerados passam sem falhas.
   - Tempo de execução aceitável (< Y segundos para N arquivos).

6. Roadmap Futuro

   - Fase 1: Melhoria de Contexto por Busca Simples
     - Enriquecer o prompt do LLM com exemplos de testes existentes, encontrados por busca de texto simples.
     - Isso aterra o modelo no estilo e nas convenções do projeto, melhorando a qualidade dos testes gerados.
   - Fase 2: RAG com Módulo de Embedding Simplificado
     - Implementar um sistema de busca semântica para encontrar os exemplos de testes mais relevantes.
     - Criar um comando para indexar os testes existentes, gerando embeddings e salvando-os localmente.
     - Utilizar a busca por similaridade de embeddings para encontrar os melhores exemplos para o prompt.
   - Suporte a múltiplos provedores de LLM.
   - Geração de mocks e fixtures automatizados.
   - Integração com pipelines de CI externos (GitHub Apps, GitLab CI).
   - Dashboard web para visualização de progressos de cobertura.

   - Fase 3: Refatoração e Simplificação de Testes Existentes
     - O LLM pode analisar testes existentes e sugerir refatorações para melhorar legibilidade, concisão e
       aplicação de padrões de teste eficazes (ex: uso inteligente de `pytest.fixture`, simplificação de
       asserções complexas, remoção de duplicação).
     - Benefício: Melhoria da manutenibilidade dos testes.

   - Fase 4: Geração Inteligente de Dados de Teste
     - O LLM pode analisar a assinatura de funções, tipos, docstrings e código-fonte para sugerir ou gerar
       dados de teste (valores para parâmetros, objetos mockados) que cubram casos de borda e diferentes
       caminhos de execução.
     - Benefício: Aumento da eficácia dos testes ao cobrir uma gama mais ampla de cenários.
