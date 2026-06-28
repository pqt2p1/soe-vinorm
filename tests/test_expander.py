from soe_vinorm.nsw_expander import RuleBasedNSWExpander


class TestRuleBasedNSWExpander:
    """Test cases for RuleBasedNSWExpander class."""

    def test_init_with_custom_dicts(self, vn_dict, abbr_dict):
        """Test expander initialization with custom dictionaries."""
        expander = RuleBasedNSWExpander(vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert expander._vn_dict == set(vn_dict)
        assert expander._abbr_dict == abbr_dict

    def test_init_with_default_dicts(self):
        """Test expander initialization with default dictionaries."""
        expander = RuleBasedNSWExpander()
        assert len(expander._vn_dict) > 0
        assert len(expander._abbr_dict) > 0

    def test_expand_with_urle_arg(self):
        """Test expander with urle."""
        expander = RuleBasedNSWExpander(expand_urle=True)
        result = expander.expand(["https://www.example.com"], ["B-URLE"])
        assert result != ["https://www.example.com"]

        expander = RuleBasedNSWExpander(expand_urle=False)
        result = expander.expand(["https://www.example.com"], ["B-URLE"])
        assert result == ["https://www.example.com"]

    def test_expand_with_sequence_arg(self):
        """Test expander with sequence."""
        expander = RuleBasedNSWExpander(expand_sequence=True)
        result = expander.expand(["abc"], ["B-LSEQ"])
        assert result != ["abc"]

        expander = RuleBasedNSWExpander(expand_sequence=False)
        result = expander.expand(["abc"], ["B-LSEQ"])
        assert result == ["abc"]

    def test_expand_sequence_keeps_foreign_name_but_expands_code(self):
        """Test proper-name-like sequences are kept while codes are expanded."""
        expander = RuleBasedNSWExpander(expand_sequence=True)

        assert expander.expand(["McMurdo", "."], ["B-LSEQ", "I-LSEQ"]) == [
            "McMurdo ."
        ]
        assert expander.expand(["P800"], ["B-LSEQ"]) != ["P800"]

    def test_expand_unknown_arg(self):
        """Test O-tagged unknown tokens can be kept unchanged."""
        expander = RuleBasedNSWExpander(expand_unknown=True)
        result = expander.expand(["OpenAI"], ["O"])
        assert result != ["OpenAI"]

        expander = RuleBasedNSWExpander(expand_unknown=False)
        result = expander.expand(["OpenAI"], ["O"])
        assert result == ["OpenAI"]

    def test_unknown_abbreviation_respects_expand_sequence_arg(self):
        """Test unknown abbreviations can be kept unchanged for downstream TTS."""
        expander = RuleBasedNSWExpander(expand_sequence=True)
        result = expander.expand(["Mantra"], ["B-LABB"])
        assert result != ["Mantra"]

        expander = RuleBasedNSWExpander(expand_sequence=False)
        result = expander.expand(["Mantra"], ["B-LABB"])
        assert result == ["Mantra"]

    def test_known_abbreviation_expands_when_sequence_disabled(self, abbr_dict):
        """Test known abbreviations still expand when unknown spell-out is disabled."""
        expander = RuleBasedNSWExpander(
            abbr_dict=abbr_dict,
            expand_sequence=False,
        )
        result = expander.expand(["ATTT"], ["B-LABB"])
        assert result == ["An toàn thông tin"]

    def test_expand_quarter(self):
        """Test valid quarter notation expansion."""
        expander = RuleBasedNSWExpander()
        result = expander.expand(["I/2024"], ["B-NQUA"])
        assert result == ["một năm hai nghìn không trăm hai mươi tư"]

    def test_expand_quarter_fallback_for_roman(self):
        """Test malformed quarter labels are kept unchanged for downstream TTS."""
        expander = RuleBasedNSWExpander()
        result = expander.expand(["XLVII"], ["B-NQUA"])
        assert result == ["XLVII"]

    def test_expand_quarter_fallback_for_malformed_text(self):
        """Test invalid quarter-like text is kept unchanged."""
        expander = RuleBasedNSWExpander()
        result = expander.expand(["Quý/abc"], ["B-NQUA"])
        assert result == ["Quý/abc"]

    def test_expand_roman_valid_numeral(self):
        """Test valid Roman numerals are expanded."""
        expander = RuleBasedNSWExpander()
        result = expander.expand(["XII"], ["B-ROMA"])
        assert result == ["mười hai"]

    def test_expand_roman_fallback_keeps_original_text(self):
        """Test malformed Roman labels keep the original token unchanged."""
        expander = RuleBasedNSWExpander()
        assert expander.expand(["KH8"], ["B-ROMA"]) == ["KH8"]
        assert expander.expand(["KH9"], ["B-ROMA"]) == ["KH9"]
        assert expander.expand(["abc123"], ["B-ROMA"]) == ["abc123"]
        assert expander.expand([""], ["B-ROMA"]) == [""]
