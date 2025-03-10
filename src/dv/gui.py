"""Modern GUI for interacting with the document Q&A system using CustomTkinter."""

# %%
import argparse
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import ipdb

# %%
from dv.config import settings
from dv.database import process_all_documents
from dv.logger import setup_logging
from dv.qa import create_qa_chain

# Add GUI font settings to config if not present
if not hasattr(settings, "GUI_FONT"):
    settings.GUI_FONT = {"size": 16, "weight": "bold"}

# Extract font settings
font_size = settings.GUI_FONT.get("size", 16)
font_weight = settings.GUI_FONT.get("weight", "bold")

# %%
logger = setup_logging(settings.log_level)


# %%
class CustomCTkEntry(ctk.CTkEntry):
    """Custom CTkEntry with word deletion support for Ctrl+Backspace."""

    def __init__(self, *args, **kwargs):
        """Initialize with additional key bindings."""
        super().__init__(*args, **kwargs)

        # Bind Ctrl+Backspace to delete word
        self.bind("<Control-BackSpace>", self._delete_word)
        self.bind(
            "<Control-w>", self._delete_word
        )  # Also common shortcut in many editors

    def _delete_word(self, event):
        """Delete the word before the cursor."""
        # Get current content and cursor position
        content = self.get()
        cursor_pos = self.index(tk.INSERT)

        if cursor_pos == 0:
            return "break"  # Nothing to delete

        # Find the start of the current word
        i = cursor_pos - 1
        while i >= 0 and not content[i].isspace():
            i -= 1

        # Delete from the start of the word to the cursor position
        word_start = i + 1
        self.delete(word_start, cursor_pos)

        return "break"  # Prevent default behaviour


# %%
class QAApplication:
    """Modern GUI application for the document Q&A system."""

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("DocuVerse Q&A")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)

        # Set appearance mode and default color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Animation variables for processing indicator
        self.processing_animation_id = None
        self.animation_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.animation_index = 0

        # Initialize the QA chain
        self.qa_chain = None
        self.initialize_qa_chain()

        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Set up the rows
        self.main_frame.grid_rowconfigure(0, weight=0)  # Top buttons
        self.main_frame.grid_rowconfigure(1, weight=1)  # Chat display
        self.main_frame.grid_rowconfigure(2, weight=0)  # Question label
        self.main_frame.grid_rowconfigure(3, weight=0)  # Question input
        self.main_frame.grid_rowconfigure(4, weight=0)  # Status bar

        # Create the UI components
        self.create_top_buttons()
        self.create_chat_display()
        self.create_question_area()
        self.create_status_bar()

        # Show welcome message
        self.update_chat(
            "System",
            "Welcome to DocuVerse Q&A! Ask questions about your documents in the input box below.",
        )

        # Set focus to question entry
        self.question_entry.focus_set()

    def create_top_buttons(self):
        """Create top button panel."""
        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Add document button
        self.add_doc_btn = ctk.CTkButton(
            button_frame,
            text="Add Document",
            command=self.add_document,
            height=36,
            font=ctk.CTkFont(size=font_size, weight=font_weight),
        )
        self.add_doc_btn.pack(side="left", padx=(0, 5))

        # Reindex button
        self.reindex_btn = ctk.CTkButton(
            button_frame,
            text="Reindex Documents",
            command=self.reindex_documents,
            height=36,
            font=ctk.CTkFont(size=font_size, weight=font_weight),
        )
        self.reindex_btn.pack(side="left", padx=5)

        # Reset conversation button
        self.reset_btn = ctk.CTkButton(
            button_frame,
            text="Reset Conversation",
            command=self.reset_conversation,
            height=36,
            font=ctk.CTkFont(size=font_size, weight=font_weight),
        )
        self.reset_btn.pack(side="left", padx=5)

    def create_chat_display(self):
        """Create chat display area."""
        # Frame for chat display
        chat_frame = ctk.CTkFrame(self.main_frame)
        chat_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=0)  # For label
        chat_frame.grid_rowconfigure(1, weight=1)  # For textbox

        # Chat label
        chat_label = ctk.CTkLabel(
            chat_frame,
            text="Conversation History:",
            font=ctk.CTkFont(size=font_size + 2, weight="bold"),
        )
        chat_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Create chat display with scrollbar
        self.chat_display = ctk.CTkTextbox(
            chat_frame, font=ctk.CTkFont(size=font_size), wrap="word"
        )
        self.chat_display.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Configure tag colors - need to use tag_config for textbox
        self.chat_display.tag_config("user", foreground="#3a7ebf")
        self.chat_display.tag_config("ai", foreground="#e74c3c")
        self.chat_display.tag_config("system", foreground="#95a5a6")
        self.chat_display.tag_config("separator", foreground="#4a4a4a")

    def create_question_area(self):
        """Create question input area."""
        # Question label
        question_label = ctk.CTkLabel(
            self.main_frame,
            text="Type your question here:",
            font=ctk.CTkFont(size=font_size + 2, weight="bold"),
        )
        question_label.grid(row=2, column=0, sticky="w", pady=(0, 5))

        # Frame for question entry and send button
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)

        # Create a custom Entry widget that handles Ctrl+Backspace
        self.question_entry = CustomCTkEntry(
            input_frame, font=ctk.CTkFont(size=font_size + 2), height=40
        )
        self.question_entry.grid(row=0, column=0, sticky="ew", padx=(10, 10))
        self.question_entry.bind("<Return>", self.on_send_click)

        # Send button
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.on_send_click,
            width=100,
            height=40,
            font=ctk.CTkFont(size=font_size, weight=font_weight),
        )
        self.send_btn.grid(row=0, column=1, sticky="e", padx=10, pady=10)

    def create_status_bar(self):
        """Create status bar."""
        # Status bar frame
        status_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a", corner_radius=0)
        status_frame.grid(row=4, column=0, sticky="ew")

        # Status text
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=font_size - 2),
            anchor="w",
            padx=10,
        )
        self.status_label.pack(fill="x", padx=5, pady=2)

    def initialize_qa_chain(self):
        """Initialize the QA chain on startup."""
        try:
            self.qa_chain = create_qa_chain()
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize Q&A system: {str(e)}")

    def update_chat(self, sender, message):
        """Add a message to the chat display."""
        # Enable text editing
        self.chat_display.configure(state="normal")

        # Format based on sender
        if sender == "User":
            self.chat_display.insert("end", "You: ", "user")
        elif sender == "AI":
            self.chat_display.insert("end", "AI: ", "ai")
        else:
            self.chat_display.insert("end", f"{sender}: ", "system")

        # Insert message
        self.chat_display.insert("end", f"{message}\n\n")

        # Add separator after AI responses
        if sender == "AI":
            self.chat_display.insert("end", "─" * 80 + "\n\n", "separator")

        # Scroll to the end and disable editing
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def on_send_click(self, event=None):
        """Handle send button click or Enter key."""
        question = self.question_entry.get().strip()

        if not question:
            # Flash the entry widget to indicate it's empty
            original_fg = self.question_entry.cget("fg_color")
            self.question_entry.configure(fg_color="#3d1a1a")
            self.root.after(
                100, lambda: self.question_entry.configure(fg_color=original_fg)
            )
            return

        # Clear the entry
        self.question_entry.delete(0, "end")

        # Update chat with user's question
        self.update_chat("User", question)

        # Disable UI elements during processing
        self.toggle_ui(False)

        # Start processing animation
        self.start_processing_animation()

        # Process query in a separate thread
        threading.Thread(target=self.process_query, args=(question,), daemon=True).start()

    def process_query(self, question):
        """Process a query in a background thread."""
        try:
            if not self.qa_chain:
                self.root.after(
                    0,
                    lambda: self.update_chat(
                        "System",
                        "Q&A system is not initialized. Please try reindexing documents.",
                    ),
                )
                self.stop_processing_animation()
                return

            # Get answer from QA chain
            answer = self.qa_chain.query(question)

            # Update UI in the main thread
            self.root.after(0, lambda: self.update_chat("AI", answer))

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: self.update_chat("System", f"Error: {str(e)}"))

        finally:
            # Stop the animation and re-enable UI elements
            self.root.after(0, self.stop_processing_animation)
            self.root.after(0, lambda: self.toggle_ui(True))
            self.root.after(0, lambda: self.question_entry.focus_set())

    def start_processing_animation(self):
        """Start animated processing indicator."""
        # Cancel any existing animation
        self.stop_processing_animation()

        # Start new animation
        self._update_processing_animation()

    def _update_processing_animation(self):
        """Update the processing animation frame."""
        # Get current animation character
        char = self.animation_chars[self.animation_index]

        # Update status text
        self.status_label.configure(text=f"Processing query {char}")

        # Increment animation index
        self.animation_index = (self.animation_index + 1) % len(self.animation_chars)

        # Schedule next update
        self.processing_animation_id = self.root.after(
            100, self._update_processing_animation
        )

    def stop_processing_animation(self):
        """Stop the processing animation."""
        if self.processing_animation_id:
            self.root.after_cancel(self.processing_animation_id)
            self.processing_animation_id = None
            self.status_label.configure(text="Ready")

    def toggle_ui(self, enabled):
        """Enable or disable UI elements."""
        state = "normal" if enabled else "disabled"
        self.add_doc_btn.configure(state=state)
        self.reindex_btn.configure(state=state)
        self.reset_btn.configure(state=state)
        self.send_btn.configure(state=state)
        self.question_entry.configure(state=state)

    def add_document(self):
        """Add a document to the system."""
        filetypes = [("Text files", "*.txt"), ("Markdown files", "*.md")]
        file_path = tk.filedialog.askopenfilename(
            title="Select Document", filetypes=filetypes
        )

        if not file_path:
            return

        # Copy file to books directory
        try:
            import shutil

            books_dir = Path(settings.DOCS_DIR)
            books_dir.mkdir(exist_ok=True)

            dest_path = books_dir / Path(file_path).name
            shutil.copy2(file_path, dest_path)

            self.update_chat("System", f"Document added: {Path(file_path).name}")

            # Ask user if they want to reindex
            if messagebox.askyesno("Reindex", "Would you like to reindex documents now?"):
                self.reindex_documents()

        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            messagebox.showerror("Error", f"Failed to add document: {str(e)}")

    def reindex_documents(self):
        """Reindex all documents."""
        self.toggle_ui(False)
        self.start_processing_animation()

        # Process in separate thread
        threading.Thread(target=self._reindex_documents, daemon=True).start()

    def _reindex_documents(self):
        """Reindex documents in a separate thread."""
        try:
            success = process_all_documents()

            if success:
                # Reinitialize QA chain with fresh index
                self.qa_chain = create_qa_chain()

                # Update UI in main thread
                self.root.after(
                    0,
                    lambda: self.update_chat("System", "Documents indexed successfully."),
                )
            else:
                # Update UI in main thread
                self.root.after(
                    0, lambda: self.update_chat("System", "Failed to index documents.")
                )
        except Exception as e:
            logger.error(f"Error reindexing documents: {str(e)}")
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error", f"Failed to reindex documents: {str(e)}"
                ),
            )
        finally:
            # Stop animation and re-enable UI in main thread
            self.root.after(0, self.stop_processing_animation)
            self.root.after(0, lambda: self.toggle_ui(True))
            self.root.after(0, lambda: self.question_entry.focus_set())

    def reset_conversation(self):
        """Reset the conversation history."""
        if not self.qa_chain:
            messagebox.showerror("Error", "Q&A system is not initialized")
            return

        self.qa_chain.reset_chat_history()

        # Clear chat display
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

        # Add welcome message
        self.update_chat("System", "Conversation has been reset.")

        # Focus on entry
        self.question_entry.focus_set()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GUI for interactive Q&A system for documents"
    )
    parser.add_argument(
        "--model", type=str, default=settings.LLM_MODEL, help="Ollama model to use"
    )
    parser.add_argument(
        "--light-mode", action="store_true", help="Use light mode instead of dark mode"
    )
    return parser.parse_args()


def main():
    """Run the GUI application."""
    args = parse_args()

    # Update settings based on args
    if args.model:
        settings.LLM_MODEL = args.model

    # Create and run the application
    ctk.set_appearance_mode("light" if args.light_mode else "dark")
    root = ctk.CTk()
    app = QAApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
