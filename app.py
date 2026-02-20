#app.py
from __future__ import annotations

import threading
import queue
import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

from skyrim_actions import run_min_cover_actions


class QueueWriter:
    """File-like object that pushes writes into a queue for the GUI."""
    def __init__(self, q: queue.Queue[str]):
        self.q = q

    def write(self, s: str) -> None:
        if s:
            self.q.put(s)

    def flush(self) -> None:
        pass


class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None

        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            justify="left",
            wraplength=300,
        )
        label.pack(ipadx=6, ipady=4)

    def hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def main():
    root = tk.Tk()
    root.title("Skyrim Minimum Cover (Potions + Eat)")
    root.geometry("900x560")

    q: queue.Queue[str] = queue.Queue()
    csv_path = tk.StringVar(value="")
    out_dir = tk.StringVar(value="")

    # ---- top controls
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    ttk.Label(top, text="Ingredients CSV:").pack(side="left")

    entry = ttk.Entry(top, textvariable=csv_path)
    entry.pack(side="left", fill="x", expand=True, padx=8)

    def browse():
        path = filedialog.askopenfilename(
            title="Select skyrim_ingredients_base.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            csv_path.set(path)

    ttk.Button(top, text="Browse…", command=browse).pack(side="left")

    out_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    out_frame.pack(fill="x")

    ttk.Label(out_frame, text="Output folder (optional):").pack(side="left")

    out_entry = ttk.Entry(out_frame, textvariable=out_dir)
    out_entry.pack(side="left", fill="x", expand=True, padx=8)

    ToolTip(out_entry, "If left blank, output is saved in the same directory as the input CSV.")

    def browse_out():
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            out_dir.set(path)

    ttk.Button(out_frame, text="Browse…", command=browse_out).pack(side="left")


    # ---- log area
    log_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    log_frame.pack(fill="both", expand=True)

    text = tk.Text(log_frame, wrap="word", state="disabled")
    text.pack(side="left", fill="both", expand=True)

    scroll = ttk.Scrollbar(log_frame, command=text.yview)
    scroll.pack(side="right", fill="y")
    text.config(yscrollcommand=scroll.set)

    def append_log(s: str) -> None:
        text.configure(state="normal")
        text.insert("end", s)
        text.see("end")
        text.configure(state="disabled")

    
    progress = ttk.Progressbar(root, mode="indeterminate")
    progress.pack(fill="x", padx=10, pady=(0, 10))

    # ---- run button
    def run():
        path = csv_path.get().strip()
        if not path:
            messagebox.showerror("Missing file", "Please select a CSV file.")
            return
        if not os.path.exists(path):
            messagebox.showerror("File not found", "That CSV path does not exist.")
            return

        run_btn.config(state="disabled")
        entry.config(state="disabled")
        append_log("Starting…\n")

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = QueueWriter(q)
            try:
                out_path = run_min_cover_actions(path,out_dir.get().strip() or None)
                print(f"Output saved to: {out_path}")
                print("Done.")
            except Exception as e:
                print("\nERROR:", repr(e))
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                q.put("__ENABLE_UI__")

        threading.Thread(target=worker, daemon=True).start()
        progress.start(10)
    

    run_btn = ttk.Button(top, text="Run", command=run)
    run_btn.pack(side="left", padx=8)


    # ---- pump print output queue into the Text widget
    def pump():
        try:
            while True:
                s = q.get_nowait()
                if s == "__ENABLE_UI__":
                    progress.stop() 
                    run_btn.config(state="normal")
                    entry.config(state="normal")
                    continue
                append_log(s)
        except queue.Empty:
            pass
        root.after(50, pump)

    pump()
    root.mainloop()


if __name__ == "__main__":
    main()