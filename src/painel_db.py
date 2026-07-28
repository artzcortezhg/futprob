# -*- coding: utf-8 -*-
"""
Schema e helpers do SQLite usados pelo painel (dashboard.py) e, mais pra
frente, pelo bot/coletor: tabela de registros (apostas de papel, abertas e
fechadas, com EV/CLV), log de coletas (Betano/Superbet/OddsPapi) e contador
de uso da OddsPapi. Reaproveita o mesmo banco de predict.py
(db/previsoes.sqlite) como fonte única de verdade.

Todas as funções aqui são either (a) leitura, ou (b) inserção/atualização
explícita e auditável — nada apaga registros. A única "escrita manual"
permitida pelo painel é marcar o resultado de um jogo já ocorrido
(marcar_resultado_manual), que só ATUALIZA o status/resultado/CLV de um
registro existente, nunca remove linhas.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DB_PADRAO = RAIZ / "db" / "previsoes.sqlite"

SQL_CRIAR_TABELAS_PAINEL = """
CREATE TABLE IF NOT EXISTS registros (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    liga TEXT NOT NULL,
    data_jogo TEXT,
    time_casa TEXT NOT NULL,
    time_fora TEXT NOT NULL,
    mercado TEXT NOT NULL,
    selecao TEXT NOT NULL,
    casa_apostas TEXT,
    prob_modelo REAL,
    odd_registrada REAL,
    ev REAL,
    odd_fechamento REAL,
    clv REAL,
    status TEXT NOT NULL DEFAULT 'aberto',
    resultado TEXT,
    origem TEXT NOT NULL DEFAULT 'manual',
    apostaria INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS coletas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    executado_em TEXT NOT NULL,
    fonte TEXT NOT NULL,
    tipo TEXT,
    sucesso INTEGER NOT NULL,
    mensagem TEXT,
    n_jogos_capturados INTEGER,
    n_mercados_capturados INTEGER
);

CREATE TABLE IF NOT EXISTS oddspapi_uso (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    executado_em TEXT NOT NULL,
    endpoint TEXT,
    sucesso INTEGER NOT NULL,
    observacao TEXT
);

CREATE TABLE IF NOT EXISTS bot_estado (
    chave TEXT PRIMARY KEY,
    valor TEXT
);

CREATE TABLE IF NOT EXISTS bot_mensagens_jogo (
    message_id INTEGER PRIMARY KEY,
    criado_em TEXT NOT NULL,
    liga TEXT NOT NULL,
    time_casa TEXT NOT NULL,
    time_fora TEXT NOT NULL,
    data_jogo TEXT,
    probs_json TEXT NOT NULL
);
"""


def inicializar_db_painel(caminho_db: Path = CAMINHO_DB_PADRAO) -> None:
    caminho_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(caminho_db) as conn:
        conn.executescript(SQL_CRIAR_TABELAS_PAINEL)
        # migração: bancos criados antes da coluna 'apostaria' existir
        colunas = {linha[1] for linha in conn.execute("PRAGMA table_info(registros)")}
        if "apostaria" not in colunas:
            conn.execute("ALTER TABLE registros ADD COLUMN apostaria INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def inserir_registro(
    caminho_db: Path,
    liga: str, time_casa: str, time_fora: str, mercado: str, selecao: str,
    prob_modelo: float, odd_registrada: float, ev: float,
    casa_apostas: str | None = None, data_jogo: str | None = None,
    origem: str = "manual",
    apostaria: bool = False,
) -> int:
    """Grava um novo registro (aposta de papel) no momento em que é gerado.
    `apostaria=True` marca que esse registro em particular passou dos
    guarda-corpos de EV (src/guardrails.py) — usado pela seção 'Apostarias
    de hoje' do painel pra distinguir de registros que só existem para
    estudo (todo candidato do ranking é registrado, não só a apostaria)."""
    inicializar_db_painel(caminho_db)
    criado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        cursor = conn.execute(
            """INSERT INTO registros
               (criado_em, liga, data_jogo, time_casa, time_fora, mercado, selecao,
                casa_apostas, prob_modelo, odd_registrada, ev, status, origem, apostaria)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'aberto', ?, ?)""",
            (criado_em, liga, data_jogo, time_casa, time_fora, mercado, selecao,
             casa_apostas, prob_modelo, odd_registrada, ev, origem, int(apostaria)),
        )
        conn.commit()
        return cursor.lastrowid


def fechar_registro(caminho_db: Path, registro_id: int, odd_fechamento: float, clv: float, resultado: str | None = None) -> None:
    """Atualiza um registro existente com o fechamento (CLV) e, se souber,
    o resultado da partida. Nunca remove a linha."""
    inicializar_db_painel(caminho_db)
    with sqlite3.connect(caminho_db) as conn:
        conn.execute(
            "UPDATE registros SET odd_fechamento=?, clv=?, status='fechado', resultado=COALESCE(?, resultado) WHERE id=?",
            (odd_fechamento, clv, resultado, registro_id),
        )
        conn.commit()


def marcar_resultado_manual(caminho_db: Path, registro_id: int, resultado: str) -> None:
    """Única ação de escrita disponível no painel: marcar manualmente o
    resultado ('ganhou'/'perdeu') de um jogo já ocorrido. Não apaga nada."""
    if resultado not in ("ganhou", "perdeu"):
        raise ValueError("resultado deve ser 'ganhou' ou 'perdeu'.")
    inicializar_db_painel(caminho_db)
    with sqlite3.connect(caminho_db) as conn:
        conn.execute("UPDATE registros SET resultado=? WHERE id=?", (resultado, registro_id))
        conn.commit()


def registrar_coleta(
    caminho_db: Path, fonte: str, sucesso: bool, mensagem: str = "",
    tipo: str | None = None, n_jogos_capturados: int = 0, n_mercados_capturados: int = 0,
) -> int:
    inicializar_db_painel(caminho_db)
    executado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        cursor = conn.execute(
            """INSERT INTO coletas (executado_em, fonte, tipo, sucesso, mensagem, n_jogos_capturados, n_mercados_capturados)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (executado_em, fonte, tipo, int(sucesso), mensagem, n_jogos_capturados, n_mercados_capturados),
        )
        conn.commit()
        return cursor.lastrowid


def registrar_uso_oddspapi(caminho_db: Path, endpoint: str, sucesso: bool, observacao: str = "") -> int:
    inicializar_db_painel(caminho_db)
    executado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        cursor = conn.execute(
            "INSERT INTO oddspapi_uso (executado_em, endpoint, sucesso, observacao) VALUES (?, ?, ?, ?)",
            (executado_em, endpoint, int(sucesso), observacao),
        )
        conn.commit()
        return cursor.lastrowid


def salvar_estado_bot(caminho_db: Path, chave: str, valor: str) -> None:
    inicializar_db_painel(caminho_db)
    with sqlite3.connect(caminho_db) as conn:
        conn.execute("INSERT INTO bot_estado (chave, valor) VALUES (?, ?) ON CONFLICT(chave) DO UPDATE SET valor=excluded.valor", (chave, valor))
        conn.commit()


def carregar_estado_bot(caminho_db: Path, chave: str) -> str | None:
    inicializar_db_painel(caminho_db)
    with sqlite3.connect(caminho_db) as conn:
        row = conn.execute("SELECT valor FROM bot_estado WHERE chave=?", (chave,)).fetchone()
    return row[0] if row else None


def salvar_mensagem_jogo(caminho_db: Path, message_id: int, liga: str, time_casa: str, time_fora: str,
                          data_jogo: str | None, probs_json: str) -> None:
    """Cacheia, por message_id, qual jogo e quais probabilidades o bot
    mandou — usado quando o usuário RESPONDE a essa mensagem colando odds."""
    inicializar_db_painel(caminho_db)
    criado_em = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(caminho_db) as conn:
        conn.execute(
            """INSERT INTO bot_mensagens_jogo (message_id, criado_em, liga, time_casa, time_fora, data_jogo, probs_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET probs_json=excluded.probs_json""",
            (message_id, criado_em, liga, time_casa, time_fora, data_jogo, probs_json),
        )
        conn.commit()


def carregar_mensagem_jogo(caminho_db: Path, message_id: int) -> dict | None:
    inicializar_db_painel(caminho_db)
    with sqlite3.connect(caminho_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM bot_mensagens_jogo WHERE message_id=?", (message_id,)).fetchone()
    return dict(row) if row else None
