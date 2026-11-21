"""Pick the trials of an earlier session to record again (makeup recording).

Runs the per-trial quality check (studio.record.quality) on the chosen session and lists
every trial with missing EMG or a signal that does not match its gesture (checked), and
borderline ones (unchecked). Any other trial can be added with "모든 시도 보기".
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..record.quality import SessionCheck, TrialCheck
from .theme import MUTED

BAD_FG, DOUBT_FG = "#ff8787", "#ffd43b"


class MakeupDialog(QDialog):
    def __init__(self, sc: SessionCheck, parent=None) -> None:
        super().__init__(parent)
        self.sc = sc
        self.setWindowTitle("보충 녹화할 시도 고르기")
        self.resize(860, 560)
        v = QVBoxLayout(self)
        m = sc.meta
        head = QLabel(f"<b>{m.get('participant', '')}</b> · {m.get('created_local', '')} · "
                      f"동작 세트 {m.get('gesture_set', 'basic')}<br>"
                      f"시도 {len(sc.trials)}개 중 보충 필요 {len(sc.bad)}개, "
                      f"애매 {sum(1 for c in sc.trials if c.doubts and not c.bad)}개. "
                      f"정상 시도끼리 교차 분류 정확도 {sc.cv_accuracy:.2f}")
        head.setWordWrap(True)
        v.addWidget(head)
        expl = QLabel("빨강 = EMG 누락이나 다른 동작처럼 보이는 신호 (기본 선택), 노랑 = 애매함 (영상으로 확인 후 선택). "
                      "신호 판정은 같은 세션의 다른 시도와 비교한 것이라, 같은 동작이 여러 번 잘못됐으면 놓칠 수 있습니다.")
        expl.setWordWrap(True)
        expl.setStyleSheet(f"color: {MUTED};")
        v.addWidget(expl)
        for note in sc.notes:
            v.addWidget(QLabel(note))

        self.show_all = QCheckBox("모든 시도 보기")
        self.show_all.toggled.connect(self._filter)
        v.addWidget(self.show_all)
        self.table = QTableWidget(len(sc.trials), 5)
        self.table.setHorizontalHeaderLabels(["보충", "순서", "동작", "본인 동작 비율", "이유"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r, c in enumerate(sc.trials):
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            box.setCheckState(Qt.Checked if c.bad else Qt.Unchecked)
            self.table.setItem(r, 0, box)
            cells = [f"세트{c.set + 1} #{c.trial + 1}", f"{sc.names.get(c.label, c.label)} ({c.rep + 1}회차)",
                     "-" if c.own != c.own else f"{100 * c.own:.0f}%",
                     " / ".join(c.reasons + c.doubts) or "정상"]
            for k, text in enumerate(cells, 1):
                it = QTableWidgetItem(text)
                if c.bad or c.doubts:
                    it.setForeground(QColor(BAD_FG if c.bad else DOUBT_FG))
                self.table.setItem(r, k, it)
        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(self._count)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.count = QLabel()
        row.addWidget(self.count, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("이 시도들로 보충 녹화")
        bb.button(QDialogButtonBox.Cancel).setText("취소")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        self.ok = bb.button(QDialogButtonBox.Ok)
        row.addWidget(bb)
        v.addLayout(row)
        self._filter()
        self._count()

    def _filter(self) -> None:
        for r, c in enumerate(self.sc.trials):
            self.table.setRowHidden(r, not (self.show_all.isChecked() or c.bad or c.doubts))

    def _count(self) -> None:
        n = len(self.selected())
        self.count.setText(f"선택 {n}개")
        self.ok.setEnabled(n > 0)

    def selected(self) -> list[TrialCheck]:
        return [c for r, c in enumerate(self.sc.trials) if self.table.item(r, 0).checkState() == Qt.Checked]
