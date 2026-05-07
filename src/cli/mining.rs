use std::io;

use mining::{Mode, OutputSink, default_config_for_mode, run_auto_tuned_with_config_and_cancel};

use crate::runtime::resolve_data_file_path;
use crate::ui;

use super::prompt::prompt_choice;
use super::render_menu_page;

pub(super) fn show_mining_menu() -> bool {
    loop {
        render_menu_page();
        let Ok(choice) = prompt_choice(
            &[
                "1. 先挖邀请码再挖余额码",
                "2. 先挖余额码再挖邀请码",
                "3. 只挖邀请码",
                "4. 只挖余额码",
                "5. 返回上一级菜单",
                "6. 退出脚本",
            ],
            "请输入选项 (1/2/3/4/5/6): ",
            &["1", "2", "3", "4", "5", "6"],
            "1、2、3、4、5 或 6",
            Some("5"),
        ) else {
            return false;
        };
        match choice.as_str() {
            "1" if show_mining_runtime_menu(Mode::InviteThenBalance) => return true,
            "1" => {}
            "2" if show_mining_runtime_menu(Mode::BalanceThenInvite) => return true,
            "2" => {}
            "3" if show_mining_runtime_menu(Mode::InviteOnly) => return true,
            "3" => {}
            "4" if show_mining_runtime_menu(Mode::BalanceOnly) => return true,
            "4" => {}
            "5" => return false,
            "6" => return true,
            _ => {}
        }
    }
}

fn show_mining_runtime_menu(mode: Mode) -> bool {
    let invite_output_file = resolve_data_file_path("mining/invite-codes.txt");
    let balance_output_file = resolve_data_file_path("mining/balance-codes.txt");
    let benchmark_cache_file = resolve_data_file_path("mining/benchmark-cache.json");
    let result =
        ui::run_with_escape_interrupt("自动挖矿运行中。", None, move |cancel_flag, log| {
            let mut config = default_config_for_mode(mode);
            config.invite_output_file = invite_output_file.clone();
            config.balance_output_file = balance_output_file.clone();
            config.benchmark_cache_file = benchmark_cache_file.clone();
            config.output = OutputSink::new(move |line| log.line(line));
            run_auto_tuned_with_config_and_cancel(config, &cancel_flag).map_err(io::Error::other)
        });
    match result {
        Ok(Some(())) | Ok(None) => false,
        Err(error) => {
            println!("自动挖矿运行失败：{}", error);
            false
        }
    }
}
