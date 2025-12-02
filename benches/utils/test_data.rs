//! Test data generators for realistic benchmarking scenarios.

use std::sync::OnceLock;

/// Cached test data to avoid regeneration overhead
static CACHED_DATA: OnceLock<TestDataCache> = OnceLock::new();

pub struct TestDataCache {
    pub emails_1kb: String,
    pub emails_10kb: String,
    pub emails_100kb: String,
    pub urls_1kb: String,
    pub urls_10kb: String,
    pub urls_100kb: String,
    pub logs_1kb: String,
    pub logs_10kb: String,
    pub logs_100kb: String,
    pub json_1kb: String,
    pub json_10kb: String,
    pub json_100kb: String,
    pub ips_1kb: String,
    pub ips_10kb: String,
    pub ips_100kb: String,
    pub html_1kb: String,
    pub html_10kb: String,
    pub html_100kb: String,
    pub text_1kb: String,
    pub text_10kb: String,
    pub text_100kb: String,
    pub code_1kb: String,
    pub code_10kb: String,
    pub code_100kb: String,
}

/// Get or initialize the test data cache
pub fn get_test_data() -> &'static TestDataCache {
    CACHED_DATA.get_or_init(|| TestDataCache {
        // Email test data
        emails_1kb: generate_email_data(1024),
        emails_10kb: generate_email_data(10 * 1024),
        emails_100kb: generate_email_data(100 * 1024),

        // URL test data
        urls_1kb: generate_url_data(1024),
        urls_10kb: generate_url_data(10 * 1024),
        urls_100kb: generate_url_data(100 * 1024),

        // Log test data
        logs_1kb: generate_log_data(1024),
        logs_10kb: generate_log_data(10 * 1024),
        logs_100kb: generate_log_data(100 * 1024),

        // JSON test data
        json_1kb: generate_json_data(1024),
        json_10kb: generate_json_data(10 * 1024),
        json_100kb: generate_json_data(100 * 1024),

        // IP address test data
        ips_1kb: generate_ip_data(1024),
        ips_10kb: generate_ip_data(10 * 1024),
        ips_100kb: generate_ip_data(100 * 1024),

        // HTML test data
        html_1kb: generate_html_data(1024),
        html_10kb: generate_html_data(10 * 1024),
        html_100kb: generate_html_data(100 * 1024),

        // Plain text test data
        text_1kb: generate_text_data(1024),
        text_10kb: generate_text_data(10 * 1024),
        text_100kb: generate_text_data(100 * 1024),

        // Code test data
        code_1kb: generate_code_data(1024),
        code_10kb: generate_code_data(10 * 1024),
        code_100kb: generate_code_data(100 * 1024),
    })
}

fn generate_email_data(target_size: usize) -> String {
    let valid_emails = [
        "john.doe@example.com",
        "jane_smith@company.org",
        "test+tag@subdomain.example.co.uk",
        "user123@test-domain.com",
        "contact@mysite.io",
        "admin@localhost.localdomain",
    ];

    let invalid_emails = [
        "not-an-email",
        "@example.com",
        "missing@domain",
        "spaces in@email.com",
        "double@@at.com",
    ];

    let mut result = String::with_capacity(target_size);
    let mut toggle = true;

    while result.len() < target_size {
        if toggle {
            result.push_str(valid_emails[result.len() % valid_emails.len()]);
        } else {
            result.push_str(invalid_emails[result.len() % invalid_emails.len()]);
        }
        result.push_str(" some filler text to pad the data ");
        toggle = !toggle;
    }

    result.truncate(target_size);
    result
}

fn generate_url_data(target_size: usize) -> String {
    let urls = [
        "https://www.example.com/path/to/resource",
        "http://subdomain.test.org:8080/api/v1/users?id=123",
        "https://github.com/user/repo/blob/main/src/file.rs",
        "http://localhost:3000/admin/dashboard",
        "https://api.service.io/endpoint?key=value&foo=bar",
    ];

    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str("Check out this link: ");
        result.push_str(urls[result.len() % urls.len()]);
        result.push_str(" for more information. ");
    }

    result.truncate(target_size);
    result
}

fn generate_log_data(target_size: usize) -> String {
    let levels = ["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"];
    let messages = [
        "Request processed successfully",
        "Connection timeout after 30s",
        "Database query failed",
        "User authentication successful",
        "Cache miss for key: user_123",
    ];

    let mut result = String::with_capacity(target_size);
    let mut day = 1;
    let mut hour = 0;

    while result.len() < target_size {
        result.push_str(&format!(
            "2024-01-{:02} {:02}:30:45 [{}] {}\n",
            day,
            hour,
            levels[result.len() % levels.len()],
            messages[result.len() % messages.len()]
        ));

        hour = (hour + 1) % 24;
        if hour == 0 {
            day = (day % 28) + 1;
        }
    }

    result.truncate(target_size);
    result
}

fn generate_json_data(target_size: usize) -> String {
    let mut result = String::with_capacity(target_size);
    let mut counter = 0;

    result.push_str("{\"users\":[");

    while result.len() < target_size - 100 {
        if counter > 0 {
            result.push(',');
        }
        result.push_str(&format!(
            r#"{{"id":{},"name":"User {}","email":"user{}@example.com","status":"active","data":"Some escaped \"quoted\" text here"}}"#,
            counter, counter, counter
        ));
        counter += 1;
    }

    result.push_str("]}");
    result.truncate(target_size);
    result
}

fn generate_ip_data(target_size: usize) -> String {
    let mut result = String::with_capacity(target_size);
    let mut octet = 1;

    while result.len() < target_size {
        result.push_str(&format!(
            "Connection from 192.168.{}.{} on port 8080. ",
            (octet / 256) % 256,
            octet % 256
        ));
        octet += 1;
    }

    result.truncate(target_size);
    result
}

fn generate_html_data(target_size: usize) -> String {
    let tags = [
        "<div class=\"container\">",
        "<p>Some paragraph text</p>",
        "<a href=\"/link\">Click here</a>",
        "<span style=\"color: red;\">Red text</span>",
        "<img src=\"image.jpg\" alt=\"description\">",
        "</div>",
    ];

    let mut result = String::with_capacity(target_size);
    result.push_str("<!DOCTYPE html><html><body>");

    while result.len() < target_size - 100 {
        result.push_str(tags[result.len() % tags.len()]);
        result.push_str(" Some text content between tags. ");
    }

    result.push_str("</body></html>");
    result.truncate(target_size);
    result
}

fn generate_text_data(target_size: usize) -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "then", "runs",
        "through", "forest", "near", "river", "where", "birds", "sing", "their", "morning",
        "songs",
    ];

    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(words[result.len() % words.len()]);
        result.push(' ');
    }

    result.truncate(target_size);
    result
}

fn generate_code_data(target_size: usize) -> String {
    let code_snippets = [
        r#"let x = "hello world";"#,
        r#"const y = 'single quoted string';"#,
        r#"var z = "escaped \"quotes\" inside";"#,
        r#"String s = "Java string literal";"#,
        r#"str = 'Python string with "nested" quotes';"#,
    ];

    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(code_snippets[result.len() % code_snippets.len()]);
        result.push('\n');
    }

    result.truncate(target_size);
    result
}

/// Generate multilingual text for Unicode testing
pub fn generate_unicode_data(target_size: usize) -> String {
    let texts = vec![
        "Hello world",      // English
        "Bonjour le monde", // French
        "Hola mundo",       // Spanish
        "Привет мир",       // Russian
        "こんにちは世界",   // Japanese
        "你好世界",         // Chinese
        "مرحبا بالعالم",    // Arabic
        "שלום עולם",        // Hebrew
        "Γειά σου κόσμε",   // Greek
    ];

    let mut result = String::with_capacity(target_size);

    while result.len() < target_size {
        result.push_str(texts[result.len() % texts.len()]);
        result.push_str(" • ");
    }

    result.truncate(target_size);
    result
}
