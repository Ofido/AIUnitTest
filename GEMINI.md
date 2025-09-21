# Instruções para o Gemini CLI

## Como rodar os testes

- Use o comando: `make tests`
- Certifique-se de que todas as dependências do `requirements.txt` estejam instaladas.
- Os testes estão localizados na pasta `tests/`.

## Estrutura do Projeto

- Código fonte: `src/ai_unit_test/`
- Testes: `tests/`
- Configurações: `pyproject.toml`, `requirements.txt`
- Documentação: `README.md`, `docs/`

## Convenções

- Siga o padrão mypy para Python de acordo com as regras flake8 e bandit configuradas.
- Use nomes descritivos para funções e variáveis.
- Testes devem cobrir casos normais e de borda.
- nunca de skip no pre-commit
- sempre rode comando dentro da env do projeto (source env/bin/activate)
- siga boas praticas de clean code.
- sempre pense o quanto for necessário
