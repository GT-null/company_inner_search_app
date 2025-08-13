"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
import platform    # 高橋_問題6
import csv      # 高橋_問題6
from pathlib import Path      # 高橋_問題6
import rank_bm25        # 高橋_問題6
import sudachipy        # 高橋_問題6
import sudachidict_full        # 高橋_問題6 
from typing import List     # 高橋_問題6
from langchain_community.retrievers import BM25Retriever    # 高橋_問題6
from langchain.retrievers import EnsembleRetriever  # 高橋_問題6
from langchain.schema import Document   # 高橋_問題6
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 高橋_問題6
from sudachipy import dictionary as sudachi_dictionary  # 高橋_問題6
from sudachipy import tokenizer as sudachi_tokenizer  # 高橋_問題6
import pandas as pd     # 高橋_問題6
import json     # 高橋_問題6
from datetime import datetime       # 高橋_問題6

############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex

### 高橋_問題6: Windows環境でRAGが正常動作するよう調整
def _normalize_docs_for_windows(docs: List[Document]) -> None:
    """Windows のときだけ page_content と metadata(str) を正規化する。破壊的変更。"""
    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        if isinstance(doc.metadata, dict):
            for k, v in list(doc.metadata.items()):
                if isinstance(v, str):
                    doc.metadata[k] = adjust_string(v)

### 高橋_問題6: 日本語テキスト向けに再結合品質を意識したスプリッタを作成
def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    """日本語テキスト向けに再結合品質を意識したスプリッタ。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " "]
)

### 高橋_問題6: Sudachi のトークナイザを 1 インスタンスだけ生成してクロージャで使い回す
def _build_bm25_preprocess():
    """Sudachi のトークナイザを 1 インスタンスだけ生成してクロージャで使い回す。"""
    sudachi = sudachi_dictionary.Dictionary(dict="full").create()
    # C は粗い分割で固有名詞を壊しにくい。用途に応じて A/B/C を調整。
    split_mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def preprocess_func(text: str):
        tokens = sudachi.tokenize(text, split_mode)
        # 重要: TF（頻度）を維持するため set() などで重複除去しない
        return [t.surface() for t in tokens if t.surface().strip()]

    return preprocess_func

def initialize_retriever() -> None:
    """
    日本語ドキュメント向けに最適化した Retriever 群を初期化し、Streamlit の session_state に格納する。

    作成される session_state のキー:
      - retriever: ベクター検索（Chroma + OpenAIEmbeddings）
      - keyword_retriever: BM25（Sudachi 形態素解析）
      - hybrid_retriever: EnsembleRetriever（dense/bm25 のハイブリッド）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    try:
        logger.info("Retriever初期化: 開始")

        # 3つすべて揃っていれば何もしない
        needed = {"retriever", "keyword_retriever", "hybrid_retriever"}
        if needed.issubset(st.session_state):
            logger.info("Retriever初期化: 既存のsession_stateを再利用してスキップ")
            return

        # データ読み込み
        logger.debug("データソース読み込み中...")
        docs_all: List[Document] = load_data_sources()
        logger.info("読み込みドキュメント数: %d", len(docs_all))

        # Windowsのみ正規化
        if platform.system() == "Windows":
            logger.debug("Windows検出: 文字列正規化を実施")
            _normalize_docs_for_windows(docs_all)

        # チャンク分割
        splitter = _build_text_splitter()
        splitted_docs = splitter.split_documents(docs_all)
        logger.info("チャンク数: %d", len(splitted_docs))

        # 埋め込み & ベクターストア（要求通り: OpenAIEmbeddings() そのまま / Chromaはメモリのみ）
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)

        st.session_state.retriever = db.as_retriever(
            search_kwargs={"k": ct.RETRIEVER_TOP_K}
        )
        logger.debug("dense retriever 構築完了 (k=%d)", ct.RETRIEVER_TOP_K)

        # BM25（日本語前処理）
        preprocess_func = _build_bm25_preprocess()
        texts = [d.page_content for d in splitted_docs]
        metas = [d.metadata for d in splitted_docs]
        bm25_core = BM25Retriever.from_texts(
                    texts=texts,
                    metadatas=metas,
                    preprocess_func=preprocess_func,
        )
        bm25_core.k = ct.RETRIEVER_TOP_K
        st.session_state.keyword_retriever = bm25_core
        logger.debug("bm25 retriever 構築完了 (k=%d)", bm25_core.k)

        # ハイブリッド
        st.session_state.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                st.session_state.retriever,
                st.session_state.keyword_retriever,
            ],
            weights=[1-ct.WEIGHTS_BM25, ct.WEIGHTS_BM25],
        )
        logger.info("ハイブリッド構築完了")

        logger.info("Retriever初期化: 正常終了")

    except Exception:
        logger.exception("Retriever初期化で例外発生")
        st.error("Retrieverの初期化に失敗しました。ログを確認してください。")


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # csvファイルをtxtファイルに変換
    # 入力/出力のCSVのパス
    csv_path = os.path.join(ct.RAG_TOP_FOLDER_PATH, "社員について/社員名簿.csv")  # 高橋_問題6
    out_path = os.path.join(ct.RAG_TOP_FOLDER_PATH, "社員について/社員名簿.txt")  # 高橋_問題6
    csv_to_single_row_jsonl_document(csv_path, out_path)

    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


# 高橋_問題6
def csv_to_single_row_jsonl_document(
    csv_path: str,
    out_path: str,
    primary_key: str = "社員ID",
    # 氏名列の候補（最初に見つかった列を処理）
    name_cols=("氏名（フルネーム）","氏名","フルネーム","名前"),
    # 「各行JSONの連結」に使う区切り（改行は使わない）
    row_json_delim=" || ",
    # 「キー:値の短い自然文」構成
    kv_sep=" | ",   # 複数キー:値ペアの連結
    kv_mid=": "     # キーと値の間
):
    """
    要件:
      - CSVの1行目はヘッダ。
      - 氏名（フルネーム）列は姓と名の間の空白（半角/全角/タブ）を除去。
      - 各データ行を「キー:値の短い自然文」に整形し、行ごとにJSON化。
      - それら"行JSON"を改行なしで連結し、ファイル全体で JSONL 1行 (= 単一ドキュメント) にまとめる。
        例: {"doc_type":"row_jsonl_corpus","text":"{行1のJSON} || {行2のJSON} || ...", "metadata":{...}}

    出力:
      out_path に JSONL を 1 行だけ書き出す（本文 text 内には改行を含めない）。
    """
    # --- CSV読み込み（エンコードのフォールバック） ---
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"CSV読み込みに失敗: {csv_path}. 最後のエラー: {last_err}")

    # --- 氏名（フルネーム）の空白除去 ---
    name_col_used = None
    for col in name_cols:
        if col in df.columns:
            name_col_used = col
            # 半角/全角スペース・タブ等の連続を除去
            df[col] = df[col].astype(str).str.replace(r"[ \t\u3000]+", "", regex=True)
            break

    # --- 値の整形（改行を潰し、前後空白を除去） ---
    def clean_val(v):
        if pd.isna(v):
            return ""
        return str(v).replace("\r", " ").replace("\n", " ").strip()

    columns = df.columns.tolist()
    primary_key_present = primary_key in columns

    # --- 各行→「キー:値」の短い自然文 → 行JSON 文字列 ---
    row_json_strings = []
    for _, row in df.iterrows():
        # 「キー:値 | キー:値 | ...」
        parts = []
        for col in columns:
            val = clean_val(row[col])
            if not val:
                continue
            parts.append(f"{col}{kv_mid}{val}")
        if not parts:
            continue
        text_line = kv_sep.join(parts)

        # 行メタデータ（row_id があれば付与）
        row_meta = {}
        if primary_key_present:
            rid = clean_val(row[primary_key])
            if rid:
                row_meta["row_id"] = rid

        row_obj = {
            "doc_type": "row",
            "source": os.path.basename(csv_path),
            "text": text_line,
            "metadata": row_meta
        }
        # コンパクトなJSON表記にして短く（改行・余分な空白なし）
        row_json_strings.append(json.dumps(row_obj, ensure_ascii=False, separators=(",", ":")))

    # --- 改行を使わずに "行JSON" を連結 ---
    big_text = row_json_delim.join(row_json_strings)

    # --- 単一ドキュメント（JSONL 1行） ---
    corpus_doc = {
        "doc_type": "row_jsonl_corpus",
        "source": os.path.basename(csv_path),
        "text": big_text,  # ← ここに行JSONが改行なしで連結されて入る
        "metadata": {
            "row_count": len(row_json_strings),
            "columns": columns,
            "primary_key": primary_key if primary_key_present else None,
            "name_col_used": name_col_used,
            "row_json_delim": row_json_delim,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
    }

    # --- 書き出し（JSONL 1行） ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(corpus_doc, ensure_ascii=False))
        f.write("\n")  # JSONLとしての1行末尾の改行（本文 text 内には改行なし）

    return out_path

# 使い方例:
# csv_to_single_row_jsonl_document("社員名簿.csv", "社員名簿_row_jsonl_corpus.txt")

'''
def csv_to_single_jsonl_txt_with_name_fix(
    csv_path: str,
    out_path: str,
    primary_key: str = "社員ID",
    name_cols=("氏名（フルネーム）","氏名","フルネーム","名前"),
    row_sep=" || ",   # 改行禁止なので行間の視認性用セパレータ
    kv_sep=" | ",     # キー:値ペアの連結
    kv_mid=": "       # キーと値の間
):
    """
    要件:
      - 氏名（フルネーム）列: 姓と名の間の空白（半角/全角/タブ）を除去
      - 1行目はヘッダ
      - Row-as-Doc: 各データ行を「キー:値の短い自然文」に整形
      - 各行は改行を使わず結合し、ファイル全体で1つのドキュメント（JSONL 1行）にまとめる
      - メタデータを付与（row_count, columns, primary_key, name_col_used, source, generated_at）
    """
    # エンコーディングのフォールバック
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"CSV読み込みに失敗: {csv_path}. 最後のエラー: {last_err}")

    # 主キーはメタで記録（今回はRow-as-Docのみ）
    primary_key_present = primary_key in df.columns

    # 氏名列の空白除去（候補のうち最初に見つかった列のみ）
    name_col_used = None
    for col in name_cols:
        if col in df.columns:
            name_col_used = col
            df[col] = df[col].astype(str).str.replace(r"[ \t\u3000]+", "", regex=True)
            break

    # 値の整形（改行除去・トリム）
    def clean_val(v):
        if pd.isna(v):
            return ""
        return str(v).replace("\r", " ").replace("\n", " ").strip()

    columns = df.columns.tolist()

    # 各行を「キー: 値」連結の1行テキストに
    row_texts = []
    for _, row in df.iterrows():
        parts = []
        for col in columns:
            val = clean_val(row[col])
            if not val:
                continue
            parts.append(f"{col}{kv_mid}{val}")
        if parts:
            row_texts.append(kv_sep.join(parts))

    # 改行なしで全行を結合（視認性のため row_sep を使用）
    big_text = row_sep.join(row_texts)

    # JSONL 1行のオブジェクト
    doc = {
        "doc_type": "row_corpus",
        "source": os.path.basename(csv_path),
        "text": big_text,
        "metadata": {
            "row_count": len(df),
            "columns": columns,
            "primary_key": primary_key if primary_key_present else None,
            "name_col_used": name_col_used,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
    }

    # 出力（JSONL 1行）
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False))
        f.write("\n")  # JSONLの規約上、1行末尾に改行のみ（本文には改行なし）

    return {
        "out_path": out_path,
        "rows_joined": len(row_texts),
        "name_col_used": name_col_used,
        "primary_key_present": primary_key_present
    }

# 使い方例:
# csv_to_single_jsonl_txt_with_name_fix("社員名簿.csv", "社員名簿_row_corpus_single.jsonl.txt")
'''

'''
def csv_to_row_cell_txt_with_name_fix(
    csv_path: str,
    out_path: str,
    primary_key: str = "社員ID",
    name_cols=("氏名", "氏名（フルネーム）", "フルネーム", "名前"),
    cell_cols=("性別","従業員区分","部署","役職","大学名","学部・学科")
):
    """
    仕様:
      1) 氏名（フルネーム）列の「姓」と「名」の間の空白（半角/全角/タブ等）を除去
      2) Row-as-Doc と Cell-as-Doc を1つの.txtにまとめて出力
         - Row-as-Doc: 2行目以降の各行を「キー: 値」を ' | ' で連結 → 1行出力 → 改行
         - Cell-as-Doc: 主キー(社員ID) + 指定6列の各セル → 1行出力 → 改行
         - 出力順序: Row-as-Doc 全行 → 空行1つ → Cell-as-Doc 全セル
    """
    # エンコードのフォールバック
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"CSV読み込みに失敗: {csv_path}. 最後のエラー: {last_err}")

    # 主キー確認
    if primary_key not in df.columns:
        raise KeyError(f"主キー列 '{primary_key}' が見つかりません。CSVに '{primary_key}' 列を追加してください。")

    # 氏名列の空白除去（最初に見つかった候補列のみ）
    for col in name_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[ \t\u3000]+", "", regex=True)  # 半角/全角空白・タブの連続を除去
            break

    # Cell-as-Doc 対象の実在列のみ使用
    existing_cols = [c for c in cell_cols if c in df.columns]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Row-as-Doc
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                parts.append(f"{col}: {sval}")
            if parts:
                f.write(" | ".join(parts) + "\n")

        # 空行で区切り
        f.write("\n")

        # Cell-as-Doc
        for _, row in df.iterrows():
            rid = row[primary_key]
            if pd.isna(rid):
                continue
            rid = str(rid).strip()
            if not rid:
                continue
            for col in existing_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                f.write(f"{primary_key}: {rid} | {col}: {sval}\n")

# 使い方例:
# csv_to_row_cell_txt_with_name_fix("社員名簿.csv", "社員名簿_row_and_cell_namefixed.txt")
'''

'''
def convert_csv_to_jsonl_txt(csv_path, out_path, important_cols=None, primary_key="社員ID"):
    """
    CSVを1つの.txt(JSONL)に変換:
      - Row-as-Doc: 1行→1 JSON {"doc_type":"row","doc_id":row_id,"row_id":row_id,"text":"キー:値 | ..."}
      - Cell-as-Doc: 重要列の各セル→1 JSON {"doc_type":"cell","doc_id":"row_id#列","row_id":row_id,"col":列,"raw":値,"text":"列: 値"}
    条件:
      - 1行目はヘッダ（キー）
      - 主キーは「社員ID」
      - 重要列: ["性別","従業員区分","部署","役職","大学名","学部・学科"]
    """
    if important_cols is None:
        important_cols = ["性別","従業員区分","部署","役職","大学名","学部・学科"]

    # 日本語CSVの読み込みに強い簡易フォールバック
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"CSV読み込み失敗: {csv_path}. 最後のエラー: {last_err}")

    # 主キー確認
    if primary_key not in df.columns:
        raise KeyError(f"主キー列 '{primary_key}' が見つかりません。CSVに '{primary_key}' 列を追加してください。")

    existing_cols = [c for c in important_cols if c in df.columns]
    missing_cols = [c for c in important_cols if c not in df.columns]

    # 出力
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        # ---- Row-as-Doc ----
        for _, row in df.iterrows():
            row_id = str(row[primary_key])
            kv_parts = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val) or str(val).strip() == "":
                    continue
                kv_parts.append(f"{col}: {str(val).strip()}")
            text = " | ".join(kv_parts)
            obj = {
                "doc_type": "row",
                "doc_id": row_id,
                "row_id": row_id,
                "text": text,
                "table": os.path.basename(csv_path),
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # ---- Cell-as-Doc ----
        for _, row in df.iterrows():
            row_id = str(row[primary_key])
            for col in existing_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                raw = str(val).strip()
                if raw == "":
                    continue
                obj = {
                    "doc_type": "cell",
                    "doc_id": f"{row_id}#{col}",
                    "row_id": row_id,
                    "col": col,
                    "raw": raw,
                    "text": f"{col}: {raw}",
                    "table": os.path.basename(csv_path),
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "out_path": out_path,
        "rows": len(df),
        "cell_cols_used": existing_cols,
        "cell_cols_missing": missing_cols,
    }

# 使い方例
# result = convert_csv_to_jsonl_txt("社員名簿.csv", "社員名簿_docs.txt")
# print(result)
'''



'''
    # CSVファイルを読み込み、TXTファイルに書き出す 高橋_問題6
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(output_txt, "w", encoding="utf-8", newline="\n") as f_out:
        reader = csv.reader(f_in)
        headers = next(reader)  # 1行目＝主キー（見出し）
        for row in reader:      # 2行目以降
            row = (row + [""] * (len(headers) - len(row)))[:len(headers)]  # 欠損を空文字で補完
            line = " , ".join(f" {h}:{v}" for h, v in zip(headers, row))
            f_out.write(line + "\n")
'''
