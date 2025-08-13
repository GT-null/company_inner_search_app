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

        # ハイブリッド（既定: dense=0.8, bm25=0.2）
        st.session_state.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                st.session_state.retriever,
                st.session_state.keyword_retriever,
            ],
            weights=[0.8, 0.2],
        )
        logger.info("ハイブリッド構築完了 (weights: dense=0.80, bm25=0.20)")

        logger.info("Retriever初期化: 正常終了")

    except Exception:
        logger.exception("Retriever初期化で例外発生")
        st.error("Retrieverの初期化に失敗しました。ログを確認してください。")

'''
def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,          # 高橋_問2 CHUNK_SIZEはconstants.pyに定義されている
        chunk_overlap=ct.CHUNK_OVERLAP,    # 高橋_問2 CHUNK_OVERLAPはconstants.pyに定義されている
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_TOP_K})   
    # 高橋_問1, 2:RAG検索時に取得するドキュメント数（k値）を定数として定義.定数はconstants.pyに定義されている

    # 高橋_問題6:キーワード検索用のretrieverを作成
    # 検索対象のドキュメントの内容を、形態素解析を用いて単語化
    def preprocess_func(text):
        tokenizer_obj = dictionary.Dictionary(dict="full").create()
        mode = tokenizer.Tokenizer.SplitMode.A
        tokens = tokenizer_obj.tokenize(text ,mode)
        words = [token.surface() for token in tokens]
        words = list(set(words))
        return words

    docs_for_keyword_search = []
    for doc in splitted_docs:
        docs_for_keyword_search.append(doc.page_content)

    # 形態素解析を用いて単語化したドキュメントを、キーワード検索用のリストに格納しキーワード検索用のretrieverを作成
    st.session_state.keyword_retriever = BM25Retriever.from_texts(
        docs_for_keyword_search,
        preprocess_func=preprocess_func,
        k=ct.RETRIEVER_TOP_K  # 高橋_問題6:キーワード検索時に取得するドキュメント数（k値）を定数として定義
    )

    # ハイブリッド検索用のRetrieverを作成
    st.session_state.hybrid_retriever = EnsembleRetriever(
        retrievers=[
            st.session_state.retriever,
            st.session_state.keyword_retriever
        ],
        weights=[1-ct.WEIGHTS_BM25, ct.WEIGHTS_BM25]
    )
'''

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
    csv_to_txt()        # 高橋_問題6
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


def csv_to_txt():
    # 入力CSVのパス
    input_csv = os.path.join(ct.RAG_TOP_FOLDER_PATH, "社員について/社員名簿.csv")  # 高橋_問題6

    # 出力TXTのパス（同じディレクトリに配置）
    output_txt = os.path.join(ct.RAG_TOP_FOLDER_PATH, "社員について/社員名簿.txt")  # 高橋_問題6

    # CSVファイルを読み込み、TXTファイルに書き出す 高橋_問題6
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(output_txt, "w", encoding="utf-8", newline="\n") as f_out:
        reader = csv.reader(f_in)
        headers = next(reader)  # 1行目＝主キー（見出し）
        for row in reader:      # 2行目以降
            row = (row + [""] * (len(headers) - len(row)))[:len(headers)]  # 欠損を空文字で補完
            line = " , ".join(f" {h}:{v}" for h, v in zip(headers, row))
            f_out.write(line + "\n")
