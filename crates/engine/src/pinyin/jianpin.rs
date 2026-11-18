//! Abbreviation predicates for the dictionary cascade (quanpin.md §5.1). Shuangpin shares them; the only difference is that it treats `zh`/`ch`/`sh` as initial tokens.

use super::syllables::is_intact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuerySource {
    Quanpin,
    Shuangpin,
}

fn is_shuangpin_initial_token(segment: &str) -> bool {
    matches!(segment, "zh" | "ch" | "sh")
}

/// 可独立代表一个音节的声母缩写；元音必须按完整音节处理。
pub fn is_pinyin_initial(segment: &str) -> bool {
    is_shuangpin_initial_token(segment)
        || (segment.len() == 1
            && segment.as_bytes()[0].is_ascii_lowercase()
            && !b"aeiouv".contains(&segment.as_bytes()[0]))
}

/// 混输整句必须以完整音节开头，后续每段只能是完整音节或声母，且确实包含缩写。
pub fn has_mixed_pinyin_segments(segments: &[String]) -> bool {
    segments.first().is_some_and(|segment| is_intact(segment))
        && segments.iter().any(|segment| !is_intact(segment))
        && segments
            .iter()
            .all(|segment| is_intact(segment) || is_pinyin_initial(segment))
}

/// 无手动分隔的混输尾部允许逐字母解释翘舌声母，只增加这一条切分，不枚举组合。
pub fn expand_mixed_pinyin_initials(segments: &[String]) -> Option<Vec<String>> {
    if !has_mixed_pinyin_segments(segments)
        || !segments
            .iter()
            .any(|segment| is_shuangpin_initial_token(segment))
    {
        return None;
    }
    Some(
        segments
            .iter()
            .flat_map(|segment| {
                if is_shuangpin_initial_token(segment) {
                    segment.chars().map(|letter| letter.to_string()).collect()
                } else {
                    vec![segment.clone()]
                }
            })
            .collect(),
    )
}

/// 词格必须保留一段一音节的完整词典读音；完整音节严格匹配，声母只匹配对应音节的开头。
pub fn matches_lattice_segments(key: &str, segments: &[String]) -> bool {
    let mut actual = key.split('\'');
    segments.iter().all(|expected| {
        let Some(syllable) = actual.next() else {
            return false;
        };
        is_intact(syllable)
            && if is_intact(expected) {
                syllable == expected
            } else {
                is_pinyin_initial(expected) && syllable.starts_with(expected)
            }
    }) && actual.next().is_none()
}

/// Every segment has length 1.
pub fn is_pure_jianpin(segments: &[String]) -> bool {
    segments.iter().all(|segment| segment.len() == 1)
}

/// More than one segment and some non-last segment is a single letter, or under shuangpin `zh`/`ch`/`sh` (QQ:292-306).
pub fn needs_mixed_jianpin_query(segments: &[String], source: QuerySource) -> bool {
    let Some((_, leading)) = segments.split_last() else {
        return false;
    };
    !leading.is_empty()
        && leading.iter().any(|segment| {
            segment.len() == 1
                || (source == QuerySource::Shuangpin && is_shuangpin_initial_token(segment))
        })
}

/// `zh`/`ch`/`sh` when present, else the first letter (QQ:308-319).
pub fn extract_initial_token(syllable: &str) -> &str {
    if let Some(prefix) = syllable.get(..2) {
        if is_shuangpin_initial_token(prefix) {
            return prefix;
        }
    }
    syllable.get(..1).unwrap_or("")
}

/// Whether a dictionary key matches abbreviated segments position by position (QQ:321-364).
pub fn matches_mixed_segments(key: &str, segments: &[String], source: QuerySource) -> bool {
    let mut actual_segments = key.split('\'');
    segments.iter().all(|expected| {
        let Some(actual) = actual_segments.next() else {
            return false;
        };
        if expected.is_empty() || actual.is_empty() {
            return false;
        }
        let strict_initial =
            source == QuerySource::Shuangpin && is_shuangpin_initial_token(expected);
        if expected.len() == 1 || strict_initial {
            match source {
                QuerySource::Shuangpin => extract_initial_token(actual) == expected,
                QuerySource::Quanpin => actual.as_bytes()[0] == expected.as_bytes()[0],
            }
        } else {
            actual == expected
        }
    }) && actual_segments.next().is_none()
}

/// `max(limit * 16, 128)`, saturating (QQ:366-373).
pub fn build_mixed_jianpin_scan_limit(limit: usize) -> usize {
    // The C++ saturates at INT_MAX, which also keeps the value bindable as an SQL LIMIT.
    const SATURATED: usize = i32::MAX as usize;
    if limit > SATURATED / 16 {
        SATURATED
    } else {
        (limit * 16).max(128)
    }
}

/// All segments intact (QQ:375-385).
pub fn can_match_exact_key(segments: &[String]) -> bool {
    !segments.is_empty() && segments.iter().all(|segment| is_intact(segment))
}

/// Last and single-letter segments get a `%` suffix, joined with `'` (QQ:258-274). The range prefix is this minus its final character, so an inner single letter keeps a literal `%` and matches nothing, which is what sends such input on to the mixed-jianpin step.
pub fn build_key_like_pattern(segments: &[String]) -> String {
    let last = segments.len().saturating_sub(1);
    let capacity = segments.iter().map(String::len).sum::<usize>() + segments.len();
    let mut result = String::with_capacity(capacity);
    for (index, segment) in segments.iter().enumerate() {
        if index > 0 {
            result.push('\'');
        }
        result.push_str(segment);
        if index == last || segment.len() == 1 {
            result.push('%');
        }
    }
    result
}

/// `prefix + "{"`: `{` sorts right after `z` (QQ:276-279).
pub fn key_prefix_upper_bound(prefix: &str) -> String {
    let mut result = String::with_capacity(prefix.len() + 1);
    result.push_str(prefix);
    result.push('{');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| (*part).to_owned()).collect()
    }

    #[test]
    fn jianpin_shapes() {
        assert!(is_pure_jianpin(&segments(&["n", "h"])));
        assert!(!is_pure_jianpin(&segments(&["n", "hao"])));
        assert!(needs_mixed_jianpin_query(
            &segments(&["n", "hao"]),
            QuerySource::Quanpin
        ));
        assert!(!needs_mixed_jianpin_query(
            &segments(&["ni", "h"]),
            QuerySource::Quanpin
        ));
        assert!(!needs_mixed_jianpin_query(
            &segments(&["n"]),
            QuerySource::Quanpin
        ));
        assert!(!needs_mixed_jianpin_query(
            &segments(&["zh", "guo"]),
            QuerySource::Quanpin
        ));
        assert!(needs_mixed_jianpin_query(
            &segments(&["zh", "guo"]),
            QuerySource::Shuangpin
        ));
        assert!(can_match_exact_key(&segments(&["ni", "hao"])));
        assert!(!can_match_exact_key(&segments(&["ni", "h"])));
        assert!(!can_match_exact_key(&[]));
    }

    #[test]
    fn mixed_lattice_spelling_keeps_complete_syllables_strict() {
        assert!(has_mixed_pinyin_segments(&segments(&["hen", "chou", "x"])));
        assert!(has_mixed_pinyin_segments(&segments(&[
            "ni", "h", "shi", "j"
        ])));
        assert!(!has_mixed_pinyin_segments(&segments(&["n", "h"])));
        assert!(!has_mixed_pinyin_segments(&segments(&["ni", "hao"])));
        assert!(!has_mixed_pinyin_segments(&segments(&["ni", "xyz"])));
        assert!(matches_lattice_segments(
            "hen'chou'xiang",
            &segments(&["hen", "chou", "x"])
        ));
        assert!(!matches_lattice_segments(
            "heng'chou'xiang",
            &segments(&["hen", "chou", "x"])
        ));
        assert!(!matches_lattice_segments(
            "gun'qiu",
            &segments(&["gun", "qi"])
        ));
        assert!(matches_lattice_segments(
            "gun'qiu",
            &segments(&["gun", "q"])
        ));
        assert!(!matches_lattice_segments(
            "hen'chou'x",
            &segments(&["hen", "chou", "x"])
        ));
        assert!(!matches_lattice_segments(
            "hen'chou'xiang'a",
            &segments(&["hen", "chou", "x"])
        ));
    }

    #[test]
    fn initial_tokens() {
        assert_eq!(extract_initial_token("zhong"), "zh");
        assert_eq!(extract_initial_token("zong"), "z");
        assert_eq!(extract_initial_token("a"), "a");
        assert_eq!(extract_initial_token(""), "");
    }

    #[test]
    fn mixed_segments_match_per_position() {
        let typed = segments(&["n", "hao"]);
        assert!(matches_mixed_segments(
            "ni'hao",
            &typed,
            QuerySource::Quanpin
        ));
        assert!(!matches_mixed_segments(
            "ni'hai",
            &typed,
            QuerySource::Quanpin
        ));
        assert!(!matches_mixed_segments(
            "ni'hao'a",
            &typed,
            QuerySource::Quanpin
        ));
        let typed = segments(&["z", "guo"]);
        assert!(matches_mixed_segments(
            "zhong'guo",
            &typed,
            QuerySource::Quanpin
        ));
        // Under shuangpin a lone `z` is the flat initial and must not match `zh`.
        assert!(!matches_mixed_segments(
            "zhong'guo",
            &typed,
            QuerySource::Shuangpin
        ));
        assert!(matches_mixed_segments(
            "zong'guo",
            &typed,
            QuerySource::Shuangpin
        ));
        let typed = segments(&["zh", "guo"]);
        assert!(matches_mixed_segments(
            "zhong'guo",
            &typed,
            QuerySource::Shuangpin
        ));
        assert!(!matches_mixed_segments(
            "zong'guo",
            &typed,
            QuerySource::Shuangpin
        ));
        assert!(!matches_mixed_segments(
            "zhong'guo",
            &typed,
            QuerySource::Quanpin
        ));
        assert!(!matches_mixed_segments(
            "ni''hao",
            &segments(&["n", "", "hao"]),
            QuerySource::Quanpin
        ));
    }

    #[test]
    fn like_patterns_and_bounds() {
        assert_eq!(build_key_like_pattern(&segments(&["ni", "hao"])), "ni'hao%");
        assert_eq!(build_key_like_pattern(&segments(&["n", "hao"])), "n%'hao%");
        assert_eq!(build_key_like_pattern(&segments(&["n", "h"])), "n%'h%");
        assert_eq!(build_key_like_pattern(&[]), "");
        assert_eq!(key_prefix_upper_bound("ni'hao"), "ni'hao{");
        let prefix = "ping'guo'ji'hao";
        assert_eq!(key_prefix_upper_bound(prefix).capacity(), prefix.len() + 1);
        assert_eq!(build_mixed_jianpin_scan_limit(1), 128);
        assert_eq!(build_mixed_jianpin_scan_limit(20), 320);
        assert_eq!(
            build_mixed_jianpin_scan_limit(i32::MAX as usize),
            i32::MAX as usize
        );
        assert_eq!(
            build_mixed_jianpin_scan_limit(usize::MAX),
            i32::MAX as usize
        );
    }
}
