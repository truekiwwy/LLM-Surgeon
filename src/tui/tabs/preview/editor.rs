//! Single-line editor of the prompt. The cursor always sits on a cluster boundary (a
//! character plus the zero-width marks that follow it: accents, joiners), which stands in
//! for grapheme segmentation without a new dependency. Widths come from `unicode-width`.

use unicode_width::UnicodeWidthChar;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LineEditor {
    text: String,
    /// Byte offset in `text`, always on a cluster boundary.
    cursor: usize,
}

/// Combining marks, joiners, variation selectors: they stay with the character before.
fn is_mark(c: char) -> bool {
    c.width() == Some(0)
}

fn cols(c: char) -> usize {
    c.width().unwrap_or(0)
}

impl LineEditor {
    /// Cursor at the end of `text`.
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            cursor: text.len(),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Display columns between the start of the line and the cursor.
    pub fn cursor_col(&self) -> usize {
        self.text[..self.cursor].chars().map(cols).sum()
    }

    pub fn insert(&mut self, c: char) {
        if c.is_control() {
            return;
        }
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn prev_boundary(&self) -> usize {
        let mut start = self.cursor;
        for (i, c) in self.text[..self.cursor].char_indices().rev() {
            start = i;
            if !is_mark(c) {
                break;
            }
        }
        start
    }

    fn next_boundary(&self) -> usize {
        let mut end = 0;
        for (i, c) in self.text[self.cursor..].char_indices() {
            if i > 0 && !is_mark(c) {
                break;
            }
            end = i + c.len_utf8();
        }
        self.cursor + end
    }

    pub fn backspace(&mut self) {
        let start = self.prev_boundary();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub fn delete(&mut self) {
        let end = self.next_boundary();
        self.text.replace_range(self.cursor..end, "");
    }

    pub fn left(&mut self) {
        self.cursor = self.prev_boundary();
    }

    pub fn right(&mut self) {
        self.cursor = self.next_boundary();
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Ctrl-U: deletes everything before the cursor.
    pub fn kill_to_start(&mut self) {
        self.text.replace_range(..self.cursor, "");
        self.cursor = 0;
    }

    /// The part of the line shown in `width` columns, scrolled so the cursor stays visible
    /// (one column is kept for it), and the cursor column inside that window.
    pub fn visible(&self, width: usize) -> (String, usize) {
        let cursor = self.cursor_col();
        let mut skipped = 0;
        let mut chars = self.text.chars().peekable();
        while cursor - skipped >= width.max(1) {
            match chars.next() {
                Some(c) => skipped += cols(c),
                None => break,
            }
        }
        // A mark whose base was scrolled out would be drawn on nothing.
        while chars.next_if(|&c| is_mark(c)).is_some() {}
        let mut shown = String::new();
        let mut used = 0;
        for c in chars {
            if used + cols(c) > width {
                break;
            }
            used += cols(c);
            shown.push(c);
        }
        (shown, cursor.saturating_sub(skipped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(s: &str) -> LineEditor {
        let mut e = LineEditor::default();
        s.chars().for_each(|c| e.insert(c));
        e
    }

    #[test]
    fn insert_move_and_delete_on_multibyte_text() {
        let mut e = typed("héllo");
        assert_eq!(e.text(), "héllo");
        e.left();
        e.left();
        e.left();
        e.backspace();
        assert_eq!(e.text(), "hllo", "the accented char is removed whole");
        e.insert('é');
        e.delete();
        assert_eq!(e.text(), "hélo");
        e.home();
        e.backspace();
        e.left();
        assert_eq!(e.text(), "hélo", "nothing before the start");
        e.end();
        e.delete();
        assert_eq!(e.text(), "hélo", "nothing after the end");
    }

    #[test]
    fn combining_marks_and_wide_chars_move_as_one_cluster() {
        // "e" + U+0301 (combining acute) is one cluster, a CJK char is two columns wide.
        let mut e = typed("ae\u{301}b");
        assert_eq!(e.cursor_col(), 3);
        e.left();
        e.left();
        e.backspace();
        assert_eq!(e.text(), "e\u{301}b", "backspace removed 'a' only");
        e.right();
        e.backspace();
        assert_eq!(e.text(), "b", "base and mark leave together");
        let mut wide = typed("日本");
        assert_eq!(wide.cursor_col(), 4);
        wide.left();
        assert_eq!(wide.cursor_col(), 2);
        wide.delete();
        assert_eq!(wide.text(), "日");
    }

    #[test]
    fn ctrl_u_kills_before_cursor_only() {
        let mut e = typed("hello world");
        e.left();
        e.left();
        e.left();
        e.left();
        e.left();
        e.kill_to_start();
        assert_eq!(e.text(), "world");
        assert_eq!(e.cursor_col(), 0);
        e.insert('x');
        assert_eq!(e.text(), "xworld");
    }

    #[test]
    fn control_characters_are_ignored() {
        let mut e = typed("a");
        e.insert('\n');
        e.insert('\u{7}');
        assert_eq!(e.text(), "a");
    }

    #[test]
    fn visible_window_follows_the_cursor() {
        let e = typed("0123456789");
        assert_eq!(e.visible(20), ("0123456789".to_owned(), 10));
        let (shown, col) = e.visible(5);
        assert_eq!(shown, "6789", "scrolled so that the cursor fits");
        assert_eq!(col, 4);
        let mut moved = e.clone();
        moved.home();
        assert_eq!(moved.visible(5), ("01234".to_owned(), 0));
        let wide = typed("日本語日本語");
        let (shown, col) = wide.visible(7);
        assert!(
            shown.chars().map(cols).sum::<usize>() <= 7 && col < 7,
            "{shown} {col}"
        );
    }
}
