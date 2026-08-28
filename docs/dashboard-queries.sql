-- Metabase — "RAG Pipeline — Overview"
-- Conecte o Metabase em Host: postgres / Port: 5432 / Database: postgres
-- (a porta 5433 e apenas o mapeamento para o host, fora da rede do Docker).

-- ---------------------------------------------------------------------------
-- Card 1 — Total indexado
-- ---------------------------------------------------------------------------
SELECT COUNT(*) AS total_embeddings
FROM company_embeddings;

-- ---------------------------------------------------------------------------
-- Card 2 — Por estado (top 10)
-- UF real depende de Estabelecimentos0.zip: o arquivo Empresas da Receita nao
-- traz UF. Sem ele, todos os registros da Receita saem como 'NAO INFORMADO'.
-- ---------------------------------------------------------------------------
SELECT metadata->>'uf' AS estado,
       COUNT(*)        AS total
FROM company_embeddings
WHERE metadata->>'uf' IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- Card 3 — Por fonte
-- ---------------------------------------------------------------------------
SELECT metadata->>'source' AS fonte,
       COUNT(*)            AS total
FROM company_embeddings
GROUP BY 1
ORDER BY 2 DESC;

-- ---------------------------------------------------------------------------
-- Card 4 — Por porte
-- ATENCAO: o codigo do porte ('01', '03', '05') fica em `porte_cod`.
-- O campo `porte` guarda a descricao ("Microempresa (ME)"), entao um CASE
-- sobre `porte` cai sempre no ELSE e o card vira 100% "Nao informado".
-- ---------------------------------------------------------------------------
SELECT CASE metadata->>'porte_cod'
           WHEN '01' THEN 'Microempresa'
           WHEN '03' THEN 'EPP'
           WHEN '05' THEN 'Demais'
           ELSE 'Nao informado'
       END      AS porte,
       COUNT(*) AS total
FROM company_embeddings
GROUP BY 1
ORDER BY 2 DESC;

-- Alternativa sem CASE: o campo ja vem legivel do pipeline.
-- SELECT metadata->>'porte' AS porte, COUNT(*) AS total
-- FROM company_embeddings GROUP BY 1 ORDER BY 2 DESC;

-- ---------------------------------------------------------------------------
-- Dashboard 2 — Busca de empresas (filtros de UF / CNAE / situacao)
-- ---------------------------------------------------------------------------
SELECT metadata->>'razao_social' AS empresa,
       metadata->>'cnpj_formatado' AS cnpj,
       metadata->>'uf'          AS uf,
       metadata->>'cnae'        AS cnae,
       metadata->>'situacao'    AS situacao,
       (metadata->>'capital_social')::numeric AS capital_social
FROM company_embeddings
WHERE (metadata->>'uf'       = {{uf}}       OR {{uf}}       IS NULL)
  AND (metadata->>'cnae'     = {{cnae}}     OR {{cnae}}     IS NULL)
  AND (metadata->>'situacao' = {{situacao}} OR {{situacao}} IS NULL)
ORDER BY capital_social DESC
LIMIT 200;

-- ---------------------------------------------------------------------------
-- Dashboard 3 — Open Finance (somente participantes do diretorio do Bacen)
-- ---------------------------------------------------------------------------
SELECT metadata->>'razao_social' AS instituicao,
       metadata->>'municipio'    AS municipio,
       metadata->>'situacao'     AS status
FROM company_embeddings
WHERE metadata->>'source' = 'bacen'
ORDER BY 1;

-- ---------------------------------------------------------------------------
-- Card 5 — Participantes do Open Finance por segmento prudencial
-- O campo `Size` do diretorio do Bacen e o segmento (S1/S2, SCD/SEP, IP...),
-- guardado em metadata->>'segmento'. Nao confundir com o porte da Receita.
-- ---------------------------------------------------------------------------
SELECT COALESCE(NULLIF(metadata->>'segmento', ''), 'Nao informado') AS segmento,
       COUNT(*) AS participantes
FROM company_embeddings
WHERE metadata->>'source' = 'bacen'
GROUP BY 1
ORDER BY 2 DESC;
