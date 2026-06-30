import pytest

from soe_vinorm.nsw_expander import RuleBasedNSWExpander


def assert_expand(expander, words, labels, expected):
    assert expander.expand(words, labels) == expected


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

    def test_expand_raises_for_mismatched_words_and_tags(self):
        """Test mismatched token and label lengths are rejected."""
        expander = RuleBasedNSWExpander()

        with pytest.raises(ValueError, match="length of words"):
            expander.expand(["abc"], ["B-LSEQ", "I-LSEQ"])

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

    @pytest.mark.parametrize("abbr", ["AT-TT", "A.T.T.T"])
    def test_abbreviation_cleanup_before_lookup(self, abbr_dict, abbr):
        """Test split abbreviations can still match a joined dictionary key."""
        expander = RuleBasedNSWExpander(abbr_dict=abbr_dict)

        assert_expand(expander, [abbr], ["B-LABB"], ["An toàn thông tin"])

    def test_empty_abbreviation_keeps_empty_text(self):
        """Test empty abbreviation groups do not crash."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [""], ["B-LABB"], [""])

    def test_expand_quarter(self):
        """Test valid quarter notation expansion."""
        expander = RuleBasedNSWExpander()
        result = expander.expand(["I/2024"], ["B-NQUA"])
        assert result == ["quý một năm hai nghìn không trăm hai mươi tư"]

    def test_expand_quarter_variants(self):
        """Test quarter notation variants."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["Q4"], ["B-NQUA"]) == ["quý bốn"]
        assert expander.expand(["Q1"], ["B-NQUA"]) == ["quý một"]
        assert expander.expand(["q3"], ["B-NQUA"]) == ["quý ba"]
        assert expander.expand(["quý", "I"], ["B-NQUA", "I-NQUA"]) == ["quý một"]
        assert expander.expand(["quý", "2"], ["B-NQUA", "I-NQUA"]) == ["quý hai"]
        assert expander.expand(["Quý", "3"], ["B-NQUA", "I-NQUA"]) == ["quý ba"]
        assert expander.expand(["quý", "IV/2025"], ["B-NQUA", "I-NQUA"]) == [
            "quý bốn năm hai nghìn không trăm hai mươi lăm"
        ]
        assert expander.expand(["Q5"], ["B-NQUA"]) == ["Q5"]

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

    @pytest.mark.parametrize(
        ("words", "labels", "expected"),
        [
            (
                ["0977", "123", "456"],
                ["B-NDIG", "I-NDIG", "I-NDIG"],
                ["không chín bảy bảy một hai ba bốn năm sáu"],
            ),
            (["-12,5"], ["B-NNUM"], ["trừ mười hai phẩy năm"]),
            (["1.234"], ["B-NNUM"], ["một nghìn hai trăm ba mươi tư"]),
            (["1,234"], ["B-NNUM"], ["một phẩy hai ba bốn"]),
            (
                ["123456789"],
                ["O"],
                [
                    "một trăm hai mươi ba triệu bốn trăm năm mươi sáu nghìn "
                    "bảy trăm tám mươi chín"
                ],
            ),
        ],
    )
    def test_expand_number_and_digit_variants(self, words, labels, expected):
        """Test digit spans and common numeric formats."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, words, labels, expected)

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("7h05m09", "bảy giờ năm phút chín giây"),
            ("8h30", "tám giờ ba mươi phút"),
            ("22h", "hai mươi hai giờ"),
            ("8/10h", "tám trên mười giờ"),
            ("8-10h", "tám đến mười giờ"),
            ("8h30-10h", "tám giờ ba mươi phút đến mười giờ"),
            (
                "09:15-10:45",
                "chín giờ mười lăm phút đến mười giờ bốn mươi lăm phút",
            ),
            ("8x30", "tám ích ba mươi"),
        ],
    )
    def test_expand_time_variants(self, token, expected):
        """Test time expansion for single values and ranges."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [token], ["B-NTIM"], [expected])

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("5/8", "năm tháng tám"),
            ("12-4", "mười hai tháng bốn"),
            ("8.11", "tám tháng mười một"),
            ("1-3/5", "một đến ngày ba tháng năm"),
            ("1/5-3/5", "một tháng năm đến ngày ba tháng năm"),
            ("1.5-3.5", "một tháng năm đến ngày ba tháng năm"),
        ],
    )
    def test_expand_day_variants(self, token, expected):
        """Test day/month expansion for compact dates and ranges."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [token], ["B-NDAY"], [expected])

    def test_expand_month_iso_variants(self):
        """Test month expansion for MM/YYYY and YYYY-MM variants."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["09/2026"], ["B-NMON"]) == [
            "chín năm hai nghìn không trăm hai mươi sáu"
        ]
        assert expander.expand(["11-2025"], ["B-NMON"]) == [
            "mười một năm hai nghìn không trăm hai mươi lăm"
        ]
        assert expander.expand(["2026-11"], ["B-NMON"]) == [
            "tháng mười một năm hai nghìn không trăm hai mươi sáu"
        ]
        assert expander.expand(["2026/7"], ["B-NMON"]) == [
            "tháng bảy năm hai nghìn không trăm hai mươi sáu"
        ]
        assert expander.expand(["2024.12"], ["B-NMON"]) == [
            "tháng mười hai năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["13/abcd"], ["B-NMON"]) == ["mười ba / a bê xê đê"]

    def test_expand_month_ranges(self):
        """Test month range expansion."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["1-3/2024"], ["B-NMON"]) == [
            "một đến tháng ba năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["1/2024-3/2024"], ["B-NMON"]) == [
            "một năm hai nghìn không trăm hai mươi tư đến tháng ba "
            "năm hai nghìn không trăm hai mươi tư"
        ]

    def test_expand_date_iso_variants(self):
        """Test date expansion for Vietnamese and ISO date formats."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["09-08-2024"], ["B-NDAT"]) == [
            "chín tháng tám năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["08/03/2025"], ["B-NDAT"]) == [
            "tám tháng ba năm hai nghìn không trăm hai mươi lăm"
        ]
        assert expander.expand(["15.08.2025"], ["B-NDAT"]) == [
            "mười lăm tháng tám năm hai nghìn không trăm hai mươi lăm"
        ]
        assert expander.expand(["2026/08/15"], ["B-NDAT"]) == [
            "mười lăm tháng tám năm hai nghìn không trăm hai mươi sáu"
        ]
        assert expander.expand(["2024-12-01"], ["B-NDAT"]) == [
            "một tháng mười hai năm hai nghìn không trăm hai mươi tư"
        ]

    def test_expand_date_ranges(self):
        """Test date range expansion."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["1-3/5/2024"], ["B-NDAT"]) == [
            "một đến ngày ba tháng năm năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["1/5/2024-3/5/2024"], ["B-NDAT"]) == [
            "một tháng năm năm hai nghìn không trăm hai mươi tư đến ngày ba "
            "tháng năm năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["2024.12.01"], ["B-NDAT"]) == [
            "một tháng mười hai năm hai nghìn không trăm hai mươi tư"
        ]
        assert expander.expand(["1/5-3/5/2024"], ["B-NDAT"]) == [
            "một tháng năm đến ngày ba tháng năm năm hai nghìn không trăm hai mươi tư"
        ]

    def test_expand_version_variants(self):
        """Test version expansion for prefixed and multi-token versions."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["v1.8.0"], ["B-NVER"]) == [
            "vê một chấm tám chấm không"
        ]
        assert expander.expand(["2.4.7"], ["B-NVER"]) == [
            "hai chấm bốn chấm bảy"
        ]
        assert expander.expand(["iOS", "18.2"], ["B-NVER", "I-NVER"]) == [
            "i O Ét mười tám chấm hai"
        ]
        assert expander.expand(["v2026.04"], ["B-NVER"]) == [
            "vê hai nghìn không trăm hai mươi sáu chấm bốn"
        ]
        assert expander.expand(["5.15.182"], ["B-NVER"]) == [
            "năm chấm mười lăm chấm một trăm tám mươi hai"
        ]
        assert expander.expand(["127.0.0.1"], ["B-NVER"]) == [
            "một hai bảy chấm không chấm không chấm một"
        ]
        assert expander.expand(["IPv4", "127.0.0.1"], ["B-NVER", "I-NVER"]) == [
            "I Pê vê bốn một hai bảy chấm không chấm không chấm một"
        ]
        assert expander.expand(["Rev", "C"], ["B-NVER", "I-NVER"]) == [
            "Rờ e vê Xê"
        ]
        assert expander.expand(["V2"], ["B-NVER"]) == ["vê hai"]

    def test_expand_url_email_and_hashtag_variants(self):
        """Test URL/email expansion and hashtag pass-through."""
        expander = RuleBasedNSWExpander(expand_urle=True)

        assert expander.expand(["#BackToSchool"], ["B-URLE"]) == ["#BackToSchool"]
        assert expander.expand(["#AIWorkshop2026"], ["B-URLE"]) == [
            "#AIWorkshop2026"
        ]
        assert expander.expand(["https://pay.example.vn/abc123"], ["B-URLE"]) != [
            "https://pay.example.vn/abc123"
        ]
        assert expander.expand(["admin@demo.edu.vn"], ["B-URLE"]) != [
            "admin@demo.edu.vn"
        ]
        assert expander.expand(["duy.nguyen@gmail.com"], ["B-URLE"]) == [
            "duy chấm nguyen a còng gờ mêu chấm com"
        ]

        expander = RuleBasedNSWExpander(expand_urle=False)
        assert expander.expand(["https://pay.example.vn/abc123"], ["B-URLE"]) == [
            "https://pay.example.vn/abc123"
        ]

    def test_expand_score_variants(self):
        """Test score expansion for common separators and consecutive scores."""
        expander = RuleBasedNSWExpander()

        assert expander.expand(["2:0"], ["B-NSCR"]) == ["hai không"]
        assert expander.expand(["25-22"], ["B-NSCR"]) == ["hai mươi lăm hai mươi hai"]
        assert expander.expand(["1-1"], ["B-NSCR"]) == ["một một"]
        assert expander.expand(["11:9"], ["B-NSCR"]) == ["mười một chín"]
        assert expander.expand(["2–1"], ["B-NSCR"]) == ["hai một"]
        assert expander.expand(["3/2"], ["B-NSCR"]) == ["ba hai"]
        assert expander.expand(["6-4", "6-3"], ["B-NSCR", "B-NSCR"]) == [
            "sáu bốn",
            "sáu ba",
        ]

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("2-3", "hai đến ba"),
            ("2–3", "hai đến ba"),
            ("0,1-1", "không phẩy một đến một"),
            ("A-B", "A đến B"),
            ("A:B", "A đến B"),
            ("abc", "a bê xê"),
        ],
    )
    def test_expand_range_variants(self, token, expected):
        """Test numeric and non-numeric ranges."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [token], ["B-NRNG"], [expected])

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("-5%", "trừ năm phần trăm"),
            ("20%", "hai mươi phần trăm"),
            ("10 %", "mười phần trăm"),
            ("10-20%", "mười đến hai mươi phần trăm"),
            ("10%-20%", "mười phần trăm đến hai mươi phần trăm"),
        ],
    )
    def test_expand_percent_variants(self, token, expected):
        """Test percentage values and ranges."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [token], ["B-NPER"], [expected])

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("1/2", "một trên hai"),
            ("3:4", "ba trên bốn"),
            ("1/2/3", "một trên hai trên ba"),
        ],
    )
    def test_expand_fraction_variants(self, token, expected):
        """Test fraction expansion with supported separators."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, [token], ["B-NFRC"], [expected])

    @pytest.mark.parametrize(
        ("words", "labels", "expected"),
        [
            (["300", "yd"], ["B-MEA", "I-MEA"], ["ba trăm yd"]),
            (["82km/h"], ["B-MEA"], ["tám mươi hai ki lô mét trên giờ"]),
            (["5foobar"], ["B-MEA"], ["năm foobar"]),
            (["1.2tr"], ["B-MEA"], ["một chấm hai triệu"]),
            (["2^-3"], ["B-MEA"], ["hai mũ trừ ba"]),
            (["1m75"], ["B-MEA"], ["một mét bảy mươi lăm"]),
            (["1m7"], ["B-MEA"], ["một mét bảy"]),
            (["1m05"], ["B-MEA"], ["một mét năm"]),
            (["mg/l"], ["B-MEA"], ["mi li gam trên lít"]),
            (["2-3kg"], ["B-MEA"], ["hai đến ba ki lô gam"]),
            (["3 kg - 5 kg"], ["B-MEA"], ["ba ki lô gam đến năm ki lô gam"]),
            (["m / s"], ["B-MEA"], ["mét trên giây"]),
            (
                ["0.3", "mm/vòng"],
                ["B-MEA", "I-MEA"],
                ["không chấm ba mi li mét trên vòng"],
            ),
            (["5", "ha"], ["B-NNUM", "B-MEA"], ["năm", "héc ta"]),
            (["chào", "ha"], ["O", "B-MEA"], ["chào", "ha"]),
        ],
    )
    def test_expand_measure_variants(self, words, labels, expected):
        """Test measurements with split units, joined units, and fallback units."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, words, labels, expected)

    @pytest.mark.parametrize(
        ("words", "labels", "expected"),
        [
            (["100USD"], ["B-MONEY"], ["một trăm u ét đê"]),
            (["100", "USD"], ["B-MONEY", "I-MONEY"], ["một trăm u ét đê"]),
            (["100", "usd"], ["B-MONEY", "I-MONEY"], ["một trăm u ét đê"]),
            (["25.000đ"], ["B-MONEY"], ["hai mươi lăm nghìn đồng"]),
            (["50€"], ["B-MONEY"], ["năm mươi ơ rô"]),
            (["1.000", "VNĐ"], ["B-MONEY", "I-MONEY"], ["một nghìn việt nam đồng"]),
            (
                ["1.000,50", "đồng"],
                ["B-MONEY", "I-MONEY"],
                ["một nghìn phẩy năm không đồng"],
            ),
            (
                ["1,000.50", "USD"],
                ["B-MONEY", "I-MONEY"],
                ["một nghìn chấm năm không u ét đê"],
            ),
            (["1.000,50 đồng"], ["B-MONEY"], ["một nghìn phẩy năm không đồng"]),
            (["1,000.50 USD"], ["B-MONEY"], ["một nghìn chấm năm không u ét đê"]),
            (["2tr450"], ["B-MONEY"], ["hai triệu bốn trăm năm mươi nghìn"]),
            (["2tr"], ["B-MONEY"], ["hai triệu"]),
            (["1.2tr"], ["B-MONEY"], ["một chấm hai triệu"]),
            (["1,2tr"], ["B-MONEY"], ["một phẩy hai triệu"]),
            (["2 tr 450"], ["B-MONEY"], ["hai triệu bốn trăm năm mươi nghìn"]),
            (["100XYZ"], ["B-MONEY"], ["một trăm XYZ"]),
            (["EUR/USD"], ["B-MONEY"], ["EUR/USD"]),
        ],
    )
    def test_expand_money_variants(self, words, labels, expected):
        """Test money expansion for mapped units, lowercase units, and fallback units."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, words, labels, expected)

    @pytest.mark.parametrize(
        ("words", "labels", "expected"),
        [
            (["OpenAI"], ["B-LWRD"], ["OpenAI"]),
            (["A55"], ["B-LWRD"], ["A năm mươi lăm"]),
            (["OpenAI-GPT_4"], ["B-LWRD"], ["OpenAI GPT bốn"]),
            (["u23"], ["B-LSEQ"], ["u hai mươi ba"]),
            (["3N2Đ"], ["B-LSEQ"], ["ba ngày hai đêm"]),
            (["5D4N"], ["B-LSEQ"], ["năm ngày bốn đêm"]),
            (["ID"], ["B-LSEQ"], ["Ai Đi"]),
            (["A007"], ["B-LSEQ"], ["Ây không không bảy"]),
            (["A-B/12"], ["B-LSEQ"], ["Ây hai phừn Bi sờ lát mười hai"]),
            (["P800"], ["B-LSEQ"], ["Pi tám trăm"]),
        ],
    )
    def test_expand_word_and_sequence_variants(self, words, labels, expected):
        """Test foreign word pass-through and sequence expansion."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, words, labels, expected)

    @pytest.mark.parametrize(
        ("words", "labels", "expected"),
        [
            (["#"], ["B-URLE"], ["hát sờ"]),
            (["www.demo.org"], ["B-URLE"], ["vê kép vê kép vê kép chấm dem âu chấm âu a chi"]),
            (
                ["ftp://mirror.example.net"],
                ["B-URLE"],
                ["ép ti pi co lừn sờ lát sờ lát mi a ro a chấm i xam pi le chấm net"],
            ),
            (["abc.def"], ["O"], ["a bê xê", ".", "đê e ép"]),
        ],
    )
    def test_expand_url_and_unknown_o_edge_cases(self, words, labels, expected):
        """Test URL fallback and O-tag splitting for dotted unknown tokens."""
        expander = RuleBasedNSWExpander()

        assert_expand(expander, words, labels, expected)
