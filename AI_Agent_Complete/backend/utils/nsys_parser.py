#!/usr/bin/env python3
"""
NVIDIA Nsight Systems (nsys) 输出文件自动化解析工具
"""
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

import sqlite3
import csv
import subprocess

@dataclass
class KernelInfo:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    layer: Optional[str] = None
    grid: Optional[Tuple[int,int,int]] = None
    block: Optional[Tuple[int,int,int]] = None
    regs_per_thread: Optional[int] = None
    shared_mem: Optional[int] = None

class NsysParser:
    """Nsys 输出文件解析器"""
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.sqlite_file: Optional[Path] = None
        if self.input_file.suffix == '.sqlite':
            self.sqlite_file = self.input_file
        elif self.input_file.suffix == '.nsys-rep':
            self.sqlite_file = self.input_file.with_suffix('.sqlite')

        self.tables: List[str] = []
        self.kernels: List[KernelInfo] = []
        self.layer_kernel_rows: List[Dict[str, Union[str, float]]] = []
        self.string_map: Dict[int, str] = {}

    def parse(self) -> None:
        # 若输入是 .nsys-rep，则先导出为 .sqlite
        if self.input_file.suffix == '.nsys-rep':
            self._parse_nsys_rep()
            return
        # 直接解析 .sqlite
        if not self.sqlite_file or not Path(self.sqlite_file).exists():
            raise FileNotFoundError(f"未找到 SQLite 文件: {self.sqlite_file or self.input_file}")
        self._parse_sqlite(self.sqlite_file)

    def _parse_nsys_rep(self) -> None:
        """解析 .nsys-rep 文件（先导出为SQLite）"""
        print("📋 检测到 .nsys-rep 文件，正在导出为SQLite格式...")
        sqlite_file = self.input_file.with_suffix('.sqlite')
        cmd = [
            'nsys', 'export',
            '--type=sqlite',
            '--force-overwrite=true',
            '--output', str(sqlite_file),
            str(self.input_file)
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ 导出成功: {sqlite_file}")
            self.sqlite_file = sqlite_file
            self._parse_sqlite(sqlite_file)
        except subprocess.CalledProcessError as e:
            print(f"❌ nsys导出失败: {e.stderr}")
            print("请确保 nsys 工具已正确安装并在PATH中")
            raise
        except FileNotFoundError:
            print("❌ 未找到 nsys 命令")
            print("请安装 NVIDIA Nsight Systems 并确保 nsys 在PATH中")
            raise

    def _parse_sqlite(self, sqlite_file: Optional[Path] = None) -> None:
        conn = sqlite3.connect(str(sqlite_file))
        try:
            self.tables = self._get_table_names(conn)
            self._load_string_ids(conn)
            self._parse_cuda_kernels(conn)
            self.layer_kernel_rows = self._query_layer_kernels(conn)
            # 导出 layer_kernels.csv 到 SQLite 同目录
            out_csv = Path(str(sqlite_file)).with_name('layer_kernels.csv')
            if self.layer_kernel_rows:
                with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['layer', 'kernel_name', 'dur_ms'])
                    w.writeheader()
                    w.writerows(self.layer_kernel_rows)
        finally:
            conn.close()

    def _get_table_names(self, conn: sqlite3.Connection) -> List[str]:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [r[0] for r in cur.fetchall()]

    def _load_string_ids(self, conn: sqlite3.Connection) -> None:
        """加载 StringIds 映射：id -> value"""
        self.string_map = {}
        if 'StringIds' not in self.tables:
            print("⚠ StringIds 表不存在，无法解码名称。")
            return
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, value FROM StringIds;")
            self.string_map = dict(cur.fetchall())
            print(f"🔠 StringIds 加载成功，共 {len(self.string_map)} 条。")
        except Exception as e:
            print(f"⚠ 读取 StringIds 失败: {e}")

    def _parse_cuda_kernels(self, conn: sqlite3.Connection) -> None:
        """解析CUDA kernel信息（解码 kernel 名称 + 结构化字段）"""
        print("🧩 正在解析 CUDA kernels 并解码 kernel 名称...")
        kernel_tables = [t for t in self.tables if t.upper().startswith('CUPTI_ACTIVITY_KIND') and 'KERNEL' in t.upper()]
        if not kernel_tables:
            print("⚠ 未找到 CUPTI kernel 表。")
            return
        ktable = kernel_tables[0]
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({ktable});")
        cols = {row[1] for row in cur.fetchall()}
        if not {'start', 'end'}.issubset(cols):
            print("⚠ CUPTI kernel 表缺少 start/end 列。")
            return
        name_col = 'demangledName' if 'demangledName' in cols else ('name' if 'name' in cols else None)
        if not name_col:
            print("⚠ CUPTI kernel 表缺少名称列（demangledName/name）。")
            return

        query = f"""
        SELECT 
            {name_col} as name_id,
            start,
            end,
            (end - start) as dur_ns,
            gridX, gridY, gridZ,
            blockX, blockY, blockZ,
            registersPerThread,
            sharedMemoryExecuted
        FROM {ktable}
        ORDER BY start
        """
        cur.execute(query)
        rows = cur.fetchall()

        for r in rows:
            name_id = r[0]
            kernel_name = self.string_map.get(name_id) if isinstance(name_id, int) else (str(name_id) if name_id is not None else "Unknown Kernel")
            self.kernels.append(KernelInfo(
                name=kernel_name or "Unknown Kernel",
                start_ns=int(r[1]),
                end_ns=int(r[2]),
                duration_ns=int(r[3]),
                grid=(r[4], r[5], r[6]) if r[4] is not None else None,
                block=(r[7], r[8], r[9]) if r[7] is not None else None,
                regs_per_thread=r[10],
                shared_mem=r[11],
            ))
        print(f"🔥 解析到 {len(self.kernels)} 个 CUDA kernels（已解码名称）")

    def _query_layer_kernels(self, conn: sqlite3.Connection) -> List[Dict]:
        """三表 JOIN：NVTX_EVENTS + StringIds + CUPTI kernel，生成 layer_kernels"""
        if 'NVTX_EVENTS' not in self.tables or 'StringIds' not in self.tables:
            return []
        kernel_tables = [t for t in self.tables if t.upper().startswith('CUPTI_ACTIVITY_KIND') and 'KERNEL' in t.upper()]
        if not kernel_tables:
            return []
        ktable = kernel_tables[0]
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({ktable});")
        cols = {row[1] for row in cur.fetchall()}

        def _find_col(candidates: Tuple[str, ...], available: set) -> Optional[str]:
            lower_map = {c.lower(): c for c in available}
            for cand in candidates:
                if cand in available:
                    return cand
                if cand.lower() in lower_map:
                    return lower_map[cand.lower()]
            return None

        start_col = _find_col(("start",), cols)
        end_col = _find_col(("end",), cols)
        if not (start_col and end_col):
            print("⚠ layer_kernels 查询失败: CUPTI kernel 表不含 end/start 列")
            return []
        name_col = _find_col(("demangledName", "name"), cols)
        if not name_col:
            print("⚠ layer_kernels 查询失败: 缺少名称列")
            return []

        runtime_tables = [t for t in self.tables if t.upper().startswith('CUPTI_ACTIVITY_KIND_RUNTIME')]
        runtime_table = runtime_tables[0] if runtime_tables else None

        nvtx_cur = conn.cursor()
        nvtx_cur.execute("PRAGMA table_info(NVTX_EVENTS);")
        nvtx_cols = {row[1] for row in nvtx_cur.fetchall()}

        corr_col = _find_col(("correlationId", "correlation_id"), cols)
        nvtx_corr_col = _find_col(("correlationId", "correlation_id"), nvtx_cols)

        def _nvtx_layer_expr(alias: str = "n") -> Tuple[str, str]:
            text_id_col = _find_col(("textId", "text_id"), nvtx_cols)
            text_col = _find_col(("text",), nvtx_cols)
            if text_id_col:
                if text_col:
                    return (f"LEFT JOIN StringIds s ON {alias}.{text_id_col} = s.id", f"COALESCE({alias}.{text_col}, s.value)")
                return (f"LEFT JOIN StringIds s ON {alias}.{text_id_col} = s.id", "s.value")
            if text_col:
                return ("", f"{alias}.{text_col}")
            return ("", "'Unknown Layer'")

        runtime_corr_col = None
        runtime_start_col = None
        runtime_end_col = None
        runtime_tid_col = None
        if runtime_table:
            rt_cur = conn.cursor()
            rt_cur.execute(f"PRAGMA table_info({runtime_table});")
            rt_cols = {row[1] for row in rt_cur.fetchall()}
            runtime_corr_col = _find_col(("correlationId", "correlation_id"), rt_cols)
            runtime_start_col = _find_col(("start",), rt_cols)
            runtime_end_col = _find_col(("end",), rt_cols)
            runtime_tid_col = _find_col(("globalTid", "global_tid", "globalThreadId"), rt_cols)

        nvtx_start_col = _find_col(("start",), nvtx_cols)
        nvtx_end_col = _find_col(("end",), nvtx_cols)
        nvtx_tid_col = _find_col(("globalTid", "global_tid", "globalThreadId"), nvtx_cols)

        use_four_join = all([runtime_table, corr_col, runtime_corr_col, runtime_start_col, runtime_end_col, runtime_tid_col, nvtx_start_col, nvtx_end_col, nvtx_tid_col])

        if use_four_join:
            print("🔗 NVTX→Runtime 用时间+globalTid 对齐，再用 correlationId 关联 Kernel")
            text_join, text_expr = _nvtx_layer_expr("n")
            sql = f"""
            WITH nvtx AS (
              SELECT n.{nvtx_start_col} AS nstart, n.{nvtx_end_col} AS nend, n.{nvtx_tid_col} AS ngtid, {text_expr} AS layer
              FROM NVTX_EVENTS n
              {text_join}
              WHERE n.{nvtx_start_col} IS NOT NULL AND n.{nvtx_end_col} IS NOT NULL AND n.{nvtx_tid_col} IS NOT NULL
            ),
            rt AS (
              SELECT r.{runtime_corr_col} AS rcid, r.{runtime_start_col} AS rstart, r.{runtime_end_col} AS rend, r.{runtime_tid_col} AS rgtid
              FROM {runtime_table} r
              WHERE r.{runtime_corr_col} IS NOT NULL AND r.{runtime_start_col} IS NOT NULL AND r.{runtime_end_col} IS NOT NULL AND r.{runtime_tid_col} IS NOT NULL
            )
            SELECT
              nvtx.layer AS layer,
              COALESCE(si.value, CAST(k.{name_col} AS TEXT)) AS kernel_name,
              ROUND(((k.{end_col} - k.{start_col}))/1e6, 3) AS dur_ms
            FROM rt
            JOIN nvtx ON rt.rgtid = nvtx.ngtid AND rt.rstart >= nvtx.nstart AND rt.rend <= nvtx.nend
            JOIN {ktable} k ON k.{corr_col} = rt.rcid
            LEFT JOIN StringIds si ON k.{name_col} = si.id
            ORDER BY k.{start_col};
            """
        elif corr_col and nvtx_corr_col:
            print("🔗 使用 correlationId 三表关联 NVTX + StringIds + CUPTI kernel")
            text_join, text_expr = _nvtx_layer_expr("n")
            sql = f"""
            WITH nvtx AS (
              SELECT n.{nvtx_corr_col} AS cid, {text_expr} AS layer
              FROM NVTX_EVENTS n
              {text_join}
              WHERE n.{nvtx_corr_col} IS NOT NULL
            )
            SELECT
              nvtx.layer AS layer,
              COALESCE(si.value, CAST(k.{name_col} AS TEXT)) AS kernel_name,
              ROUND(((k.{end_col} - k.{start_col}))/1e6, 3) AS dur_ms
            FROM {ktable} k
            JOIN nvtx ON k.{corr_col} = nvtx.cid
            LEFT JOIN StringIds si ON k.{name_col} = si.id
            ORDER BY k.{start_col};
            """
        else:
            print("🔗 使用时间范围关联 NVTX_EVENTS 与 CUPTI kernels（fallback）")
            nvtx_start_col = nvtx_start_col or "start"
            nvtx_end_col = nvtx_end_col or "end"
            text_join, text_expr = _nvtx_layer_expr("n")
            sql = f"""
            WITH nvtx AS (
              SELECT n.{nvtx_start_col} AS nstart, n.{nvtx_end_col} AS nend, {text_expr} AS layer
              FROM NVTX_EVENTS n
              {text_join}
              WHERE n.{nvtx_start_col} IS NOT NULL AND n.{nvtx_end_col} IS NOT NULL
            )
            SELECT
              nvtx.layer AS layer,
              COALESCE(si.value, CAST(k.{name_col} AS TEXT)) AS kernel_name,
              ROUND(((k.{end_col} - k.{start_col}))/1e6, 3) AS dur_ms
            FROM {ktable} k
            LEFT JOIN StringIds si ON k.{name_col} = si.id
            JOIN nvtx ON k.{start_col} >= nvtx.nstart AND k.{end_col} <= nvtx.nend
            ORDER BY k.{start_col};
            """
        rows: List[Dict[str, Union[str, float]]] = []
        try:
            for layer, kname, dur_ms in cur.execute(sql):
                rows.append({'layer': layer, 'kernel_name': str(kname or ''), 'dur_ms': float(dur_ms)})
        except Exception as e:
            print(f"⚠ layer_kernels 查询失败: {e}")
            rows = []
        return rows

    def export_to_json(self, json_path: Union[str, Path]) -> Optional[str]:
        try:
            import json as _json
            data = {
                'layer_kernels': self.layer_kernel_rows,
                'kernels_preview': [ki.__dict__ for ki in self.kernels[:50]],
                'tables': self.tables
            }
            Path(json_path).write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            return str(json_path)
        except Exception:
            return None

    def export_kernel_summary_csv(self, nsys_file: str, base_path: Union[str, Path]) -> Optional[str]:
        if self.sqlite_file:
            out_csv = Path(str(self.sqlite_file)).with_name('layer_kernels.csv')
            return str(out_csv) if out_csv.exists() else None
        return None

    def parse_kernel_summary_csv(self, csv_file: Union[str, Path]) -> List[Dict]:
        p = Path(csv_file)
        if not p.exists():
            return []
        rows = []
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

class NsysAnalyzer:
    def __init__(self, parser: NsysParser):
        self.parser = parser

    def analyze(self) -> Dict:
        stats: Dict[str, Dict] = {'kernel_analysis': {}}
        total = len(self.parser.layer_kernel_rows)
        total_ms = sum(float(r.get('dur_ms', 0.0)) for r in self.parser.layer_kernel_rows)
        avg_ms = (total_ms / total) if total else 0.0
        stats['kernel_analysis'] = {
            'total_kernels': total,
            'total_kernel_time': total_ms,
            'avg_kernel_time': avg_ms,
        }
        by_layer: Dict[str, Dict[str, float]] = {}
        for r in self.parser.layer_kernel_rows:
            lay = r.get('layer') or 'Unknown'
            by_layer.setdefault(lay, {'count': 0, 'total_ms': 0.0})
            by_layer[lay]['count'] += 1
            by_layer[lay]['total_ms'] += float(r.get('dur_ms', 0.0))
        stats['by_layer'] = by_layer
        return stats
