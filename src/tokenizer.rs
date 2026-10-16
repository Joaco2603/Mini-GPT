/// Character vocabulary built from a corpus. Token id is the index in `chars`.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    pub chars: Vec<char>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        Self { chars }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>, &'static str> {
        let mut ids = Vec::new();
        for ch in text.chars() {
            let id = self
                .chars
                .iter()
                .position(|&c| c == ch)
                .ok_or("character is not in the vocabulary")?;
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, &'static str> {
        let mut out = String::new();
        for &id in ids {
            let ch = self.chars.get(id).ok_or("token id is out of vocab")?;
            out.push(*ch);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let tok = CharTokenizer::from_text("ab ba");
        let ids = tok.encode("ab ba").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "ab ba");
        assert_eq!(tok.vocab_size(), 3);
    }
}
